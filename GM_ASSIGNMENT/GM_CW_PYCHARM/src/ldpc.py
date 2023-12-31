import numpy as np


def _is_full_rank(mat):
    rank = np.linalg.matrix_rank(mat)
    is_full_rank = rank == min(mat.shape)
    return is_full_rank


def _rearrange_to_systematic_form(ref):
    """
    Rearrange given row-echelon form to systematic form `hat_H`, whereby one side has an identity matrix of shape (n-k,
    n-k) separated out from the rest of the matrix, termed `P` (in LDPC slides).
    `hat_H` should be equal to H up to a column permutation.
    :param ref: Row-echelon form of parity check matrix. As 2d array, same shape as parity check matrix, (n-k, n).
    :return: Systematic form of parity check matrix. As 2d array, same shape as parity check matrix, (n-k, n).
    """
    hat_H = ref.copy()
    n_minus_k, n = hat_H.shape
    # identity matrix we want on left side
    id_mat = np.eye(n_minus_k)
    for i in range(n_minus_k):
        # for j in range(i, n):
        for j in range(n):
            if np.array_equal(hat_H[:, j], id_mat[:, i]):
                # swap cols
                hat_H[:, [i, j]] = hat_H[:, [j, i]]
                break
    return hat_H


def _make_systematic_encoding_matrix(hat_H, n_minus_k):
    """
    Convert systematic form of parity check matrix `H_hat` into a systematic encoding matrix (`G`) which is a
    column-wise concatenation of a matrix referred to as `P` shape (n-k, k) and an identity matrix of shape (k, k).
    :param hat_H: Systematic form of parity check matrix. A row-wise concatenation of identity matrix (n-k,
    n-k) called `I_n-k` and "something else" called `P` (n-k, k). Shown as [I_n-k | P] in slides.
    :param n_minus_k: Dimensions of the identity matrix within the systematic matrix `H_hat`.
    :return: Systematic encoding matrix `G`. As 2d array of shape (n, k).
    """
    P = hat_H[:, n_minus_k:]
    k = P.shape[1]
    I_k = np.eye(k)
    G = np.concatenate((P, I_k), axis=0)
    return G.astype(int)


def _decompose_to_echelon_form(H):
    """
    Decompose the given parity check matrix into a row-echelon form by Gaussian elimination.
    :param H: Parity check matrix. As 2d array, shape is (n-k, n).
    :return: Row-echelon form of the given parity check matrix. 2d array with same shape, (n-k, n).
    """
    ref = H.copy()
    rows, cols = ref.shape
    for i in range(rows):
        if ref[i, i] == 0:
            for j in range(i + 1, rows):
                if ref[j, i] != 0:
                    # swap rows:
                    ref[[i, j], :] = ref[[j, i], :]
                    break
        for j in range(i + 1, rows):
            if ref[j, i] == 0:
                continue
            # % 2 for F2:
            ratio = (ref[j, i] / ref[i, i]) % 2
            scale_by_ratio = (ratio * ref[i, i:]) % 2
            ref[j, i:] = np.abs((ref[j, i:] - scale_by_ratio))
    return ref.astype(np.int32)


def build_systematic_encoding_matrix(H=None):
    """
    Build the systematic form of the parity check matrix (`H_hat`) and the systematic encoding matrix (`G`) from the
    given parity check matrix (`H`), if possible.
    :param H: The parity_check_matrix. As a 2d array of shape (n-k, n).
    :return: Systematic form of the parity check matrix (`H_hat`) (shape (n-k, n)), and the systematic encoding matrix
    (`G`) (shape (n, n-k)).
    """
    if H is None:
        H = np.array([[1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 1, 0]])
        assert H.shape[0] == 3
        assert H.shape[1] == 6

    n_minus_k = H.shape[0]  # number of parity check bits
    n = H.shape[1]  # size of codewords
    k = n - n_minus_k  # size of original message

    if not _is_full_rank(H):
        print("The given parity check matrix is not full-rank, so I can't decompose it to REF")
        return None

    ref = _decompose_to_echelon_form(H)
    hat_H = _rearrange_to_systematic_form(ref)
    I_part_of_hat_H = hat_H[:n_minus_k, :n_minus_k]
    I_n_k = np.eye(n_minus_k).astype(int)

    if not np.array_equal(I_part_of_hat_H, I_n_k):
        print(f'Failed to rearrange the ref into a systematic form with identity matrix on left side')
        return None

    G = _make_systematic_encoding_matrix(hat_H, n_minus_k)

    return hat_H, G


# -------------------------- LDPC: QUESTION 3 -----------------------------------------------------------------------
# Write an LDPC-decoder based on Loopy Belief Propagation for Binary Symmetric Channel.
# Specifically, write a function that receives a parity check matrix `hat_H`, a received word `y`, a noise ratio `p`
# and an optional parameter of a maximum number of iterations (with default value of 20).
# The function should return a decoded vector along with the following return code: 0 for success,
# −1 if the maximum number of iterations is reached without a successful decoding.
# -------------------------------------------------------------------------------------------------------------------


def _compute_log_likelihood_ratios(p_y):
    """
    Convert the likelihoods of 0 and likelihoods of 1 for each possible bit, to a single value per bit in terms of a
    log-likelihood ratio. Hence convert a 2d array of shape (2, len(y)) into a 1d array of shape (1, len(y)),
    while not losing information. This is done for convenience.
    If the LLR of a bit is positive, it means probability of the bit being 0 is higher than probability of it being 1.
    And vice versa.
    :param p_y: Likelihoods of 0 and likelihoods of 1 for each possible bit. 2d array of shape (2, len(y)).
    :return: Log-likelihood ratios. 1d array of shape (1, len(y)).
    """
    log_likelihood_ratio = np.log(p_y[0] / p_y[1])
    return log_likelihood_ratio


def _init_message_passing(y, p):
    """
    Initialise the variable-to-factor message values with the likelihoods of the bit being 0 or 1 based on the
    received word `y` and the noise ratio of the channel `p`.
    :param y: Word received, that we wish to decode. Column vector 1d array. (Shape of `y1` is  (1000, 1).)
    :param p: Noise ratio of the channel. Is the probability of a bit being flipped.
    :return: Initial variable-to-factor messages, which are likelihoods of each bit of the original word being 0 or 1,
    given the bit in the received word. As a 2d array of shape (word_len, 2), hence for `y1` shape is (1000, 2).
    """
    prob_flipped, prob_not_flipped = p, 1 - p
    x0_1 = np.array([[0], [1]])
    p_y_given_x = (prob_flipped ** (x0_1 - y) % 2 * prob_not_flipped ** ((x0_1 - y + 1) % 2)).squeeze()
    return p_y_given_x


def _prob_of_y_given_x(_2d_probs):
    """
    Determine the most likely bits `max_likelihood_bits` from the given 2d array of probabilities for 0 and 1 in the
    word and compute the total probability of this most likely word.
    as a product of all the most likely bits.
    :param _2d_probs: 2d array of probabilities (the message) shape (2, word_length), corresponding to probabilities
    for 0 and probabilities for 1 over each bit of the word.
    :param probs_0_1: 1d array of probabilities
    :return: The most likely bits (1, len(y)). And a probability scalar value `total_prob`.
    """
    max_likelihood_bits = np.argmax(_2d_probs, axis=0) # the index coincides with the same values 0 and 1
    max_probs = np.max(_2d_probs, axis=0)
    total_prob = np.prod(max_probs, axis=0)
    return max_likelihood_bits, total_prob


def _passes_parity_check(word_for_parity_check, hat_H):
    """
    Check the parity of the given word. Use equation hat_H @ Gt = 0, where t is the word. The generator G multiplied
    by the word t produces the codeword x. Hx = 0 for the word that has parity.
    :param word_for_parity_check:
    :param hat_H: Systematic form of the parity check matrix H.
    :return: True if the parity is 0.
    """
    n_minus_k = hat_H.shape[0]
    G = _make_systematic_encoding_matrix(hat_H, n_minus_k)
    codeword = G * word_for_parity_check
    parity_prod = hat_H @ codeword
    return parity_prod == 0


def _indicator(scalar_value):
    """
    Indicator function
    :param scalar_value:
    :return:
    """
    return 1 if scalar_value == 0 else 0

def _make_dict_of_neighbouring_variables_of_each_factor(hat_H):
    """
    Pre-compute the list of 'neighbouring' variables to each factor at start.
    (These are the connecting variables according to a factor-graph representation.
    The variables are listed by their index positions.)
    :param hat_H: Systematic form of parity check matrix.
    :return: Index positions of neighbouring variables per factor. Dict of 1d arrays of index positions,
    key is factor row number (0-indexed).
    """
    indices_of_neighbouring_variables_per_factor = dict()
    for i, factor in enumerate(hat_H):
        indices_of_neighbouring_variables_per_factor[i] = np.where(factor == 1)[0]
    return indices_of_neighbouring_variables_per_factor


def _make_dict_of_neighbouring_factors_of_each_variable(hat_H):
    """
    Pre-compute the list of 'neighbouring' factors to each variable at start.
    (These are the connecting factors according to a factor-graph representation.
    The factors are listed by their index positions.)
    :param hat_H: Systematic form of parity check matrix.
    :return: Index positions of neighbouring factors per variable. Dict of 1d arrays of index positions,
    key is factor row number (0-indexed).
    """
    indices_of_neighbouring_factors_per_variable = dict()
    for i, variable in enumerate(hat_H.T):
        indices_of_neighbouring_factors_per_variable[i] = np.where(variable == 1)[0]
    return indices_of_neighbouring_factors_per_variable


def _get_indication(y, incoming_variables, recipient_variable):
    xn = y[recipient_variable]
    sum_for_indicator = xn
    for var_i in incoming_variables:
        sum_for_indicator += y[var_i]
    return _indicator(sum_for_indicator)


def _compute_msg_for_variable(probs_msg, y, factor_i, incoming_variables, recipient_variable):
    sum_, product = 0, 0

    for i in incoming_variables:

        ind = _get_indication(y, incoming_variables, recipient_variable)
        product = ind
        all_but_recipient = [v for v in incoming_variables if v != recipient_variable]
        for var_j in all_but_recipient:
            prob = probs_msg[factor_i, var_j]
            product *= prob

        sum_ += ind * product

    return sum_


def _compute_msg_for_factor(probs_msg, y, var_i, incoming_factors, recipient_factor):
    product = probs_msg

    for i in incoming_factors:
        all_but_recipient = [v for v in incoming_factors if v != recipient_factor]

        for fac_j in all_but_recipient:
            prob = probs_msg[var_i, fac_j]
            product *= prob

    return product


def _compute_factor_to_variable_msgs(xn_to_fm, hat_H, y, neighbours_vs_per_f):
    """
    (Re)compute factor-to-variables messages using the variable-to-factor message passed to it `xn_to_fm`.
    "(i) Take a product over all the incoming messages from variables to this factor, apart from the one to which we're
    sending the updated message.
    (ii) Multiply by the factor itself.
    (iii) Sum over all the variables apart from the one we're sending the message to."
    :param xn_to_fm: Probabilities (message) for each bit of codeword, updated from preceding variable-to-factor update.
    :param hat_H: Systematic parity check matrix. 2d array of shape (factors_num, variables_num).
    :param y: Codeword, to decode.
    :param neighbours_vs_per_f: Connected variables for each factor.
    :return: Updated message.
    """
    number_of_factors, number_of_variables = hat_H.shape
    fm_to_xn = np.zeros((number_of_factors, number_of_variables))

    for factor_i, factor in enumerate(hat_H):
        neighbour_variables = neighbours_vs_per_f[factor_i]

        for neigh_var in neighbour_variables:
            msg_for_variable = _compute_msg_for_variable(probs_msg=xn_to_fm, y, factor_i,
                                                         incoming_variables=neighbour_variables,
                                                         recipient_variable=neigh_var)
            _send_msg_to_recipient_var(neigh_var, msg_for_variable)

    return fm_to_xn


def _compute_variable_to_factor_msgs(hat_H, y, p_y_given_x, fm_to_xn, neighbours_fs_per_v: dict):
    """
    (Re)compute factor-to-variables messages updating the given variable-to-factor messages.
    "(i) Take all the incoming messages apart from the one factor to which we're ging to send the message now,
    we take a product of these and then
    (ii) we also multiply by the one initial probability, p(y|x).
    (iii) We take the product of all those and we send them over."
    :param hat_H: Systematic parity check matrix. 2d array of shape (factors_num, variables_num).
    :param y: Codeword, to decode.
    :param neighbours_fs_per_v: Connected factors for each variable.
    :return: Updated message.
    """
    number_of_factors, number_of_variables = hat_H.shape
    xn_to_fm = np.zeros((number_of_factors, number_of_variables)

    for var_i, variable in enumerate(hat_H.T):
        neighbour_factors = neighbours_fs_per_v[var_i]

        for neigh_fac in neighbour_factors:
            msg_for_factor = _compute_msg_for_factor(probs_msg, y, var_i, incoming_factors=neighbour_factors,
                                                     recipient_factor=neigh_fac)

    return xn_to_fm


def _are_marginals_relatively_unchanged(fm_to_xn, previous_marginals):
    """
    (Re)compute the marginal probabilities (aka 'beliefs'), proportional to all the incoming messages...
    :return: marginals
    """
    marginals_are_relatively_unchanged = False
    new_marginals = np.prod(fm_to_xn)
    marginals_change_threshold = 0.5
    if new_marginals - previous_marginals < marginals_change_threshold:
        marginals_are_relatively_unchanged = True
    return marginals_change_threshold, new_marginals

def _compute_candidate_word(p_y_given_x, fm_to_xn):
    candidate_word = np.zeros(())
    hat_p_x_given_y = p_y_given_x * _product_of_probs_of_connected_nodes()
    return candidate_word

# `STEPS` ARE A DIRECT REFERENCE TO SLIDE#23 IN `ldpc.pdf` OF D. ADAMSKIY LECTURE SLIDES.
def decode_vector(hat_H=None, y=None, p=0.1, max_iterations=20):
    """
    Decode given word as array, using loopy belief propagation for a binary symmetric channel.
    Compute likelihood of the bits of the given word `y`.
    :param hat_H: Systematic form of the parity check matrix. Expected as 2d numpy array.
    :param y: Received word, to decode. Expected as 1d numpy array.
    :param p: Noise ratio. 0.1 by default.
    :param max_iterations: Maximum number of iterations for the loopy belief network updates.
    :return: Decoded word and return code (0 for success, −1 if the max number of iterations is reached without a
    successful decoding.
    """
    return_code = -1

    if not hat_H:
        H = np.loadtxt('../../inputs/H1.txt')  # shape (750,1000)
        hat_H = _rearrange_to_systematic_form(_decompose_to_echelon_form(H))

    if not y:
        y = np.loadtxt('../../inputs/y1.txt').reshape(-1, 1)  # shape (1000,1)

#---# STEP 1. INITIALISE CONDITIONAL PROBABILITIES OF WORD BASED ON RECEIVED WORD `y` AND PROBABILITY OF BIT FLIP `p`:
    p_y_given_x = _init_message_passing(y, p)  #
    # `p_y_given_x` IS USED IN SUBSEQUENT PROBABILITY MESSAGE PASSING UPDATES, BUT THIS INITIAL VALUE IS STORED
    # AND RE-USED IN STEPS 3 AND 4.

    # FOR CONVENIENCE COMBINE THE TWO PROBS INTO ONE:
    xn_to_fm = _compute_log_likelihood_ratios(p_y_given_x)

    # FOR EACH FACTOR AND FOR EACH VARIABLE, STORE INDICES OF ALL CONNECTED VARIABLES AND FACTORS, RESPECTIVELY:
    i_of_neighbouring_vars_per_factor = _make_dict_of_neighbouring_variables_of_each_factor(hat_H)
    i_of_neighbouring_factors_per_var = _make_dict_of_neighbouring_factors_of_each_variable(hat_H)
    candidate_word = ''
    marginals = 0

#---# STEPS 2, 3 & 4: ------------------------------------------------------------------------------------------------

    # PERFORM MESSAGE PASSING UNTIL CONVERGENCE OR FOR THE MAXIMUM NUMBER OF ITERATIONS,
    # COMPUTING THE MARGINALS AT EACH ITERATION TO DETERMINE WHETHER CONVERGED:
    for iteration in range(max_iterations):

# ------# STEP 2. (RE)COMPUTE FACTOR-TO-VARIABLE MESSAGES (PROBABILITIES) ACCORDING TO PARITY CONSTRAINTS: -----------
        fm_to_xn = _compute_factor_to_variable_msgs(hat_H, y, xn_to_fm, i_of_neighbouring_vars_per_factor)

# ------# STEP 3. (RE)COMPUTE VARIABLE-TO-FACTOR MESSAGES (PROBABILITIES) ACCORDING TO PARITY CONSTRAINTS: -----------
        xn_to_fm = _compute_variable_to_factor_msgs(hat_H, y, p_y_given_x, fm_to_xn, i_of_neighbouring_factors_per_var)

# ------# STEP 4. COMPUTE MARGINALS. IF RELATIVELY UNCHANGED, USE PROBABILITIES TO DECODE WORD TO CANDIDATE WORD. ----
        marginals_are_unchanged, marginals = _are_marginals_relatively_unchanged(p_y_given_x, marginals)

        if marginals_are_unchanged:
            candidate_word = _compute_candidate_word(p_y_given_x, fm_to_xn)

            # CHECK PARITY OF CANDIDATE WORD AND IF PASSES PARITY, STOP, RETURN WORD & 0.
            if _passes_parity_check(candidate_word, hat_H):
                return_code = 0
                break

    return candidate_word, return_code


# START CODE FROM HERE:
if __name__ == '__main__':

    decoded_word, return_code = decode_vector()


    # H = np.array([[1, 1, 1, 1, 0, 0],
    #                [0, 0, 1, 1, 0, 1],
    #                [1, 0, 0, 1, 1, 0]], dtype=np.float64)
    # _cache_global_neighbour_variables_of_each_factor(H)

    # build_systematic_encoding_matrix()
    # pass

    # H = np.loadtxt('inputs/H1.txt')
    # # ref = np.loadtxt('inputs/ref_from_oct.txt')
    # # H = np.array([[1, 1, 1, 1, 0, 0],
    # #               [0, 0, 1, 1, 0, 1],
    # #               [1, 0, 0, 1, 1, 0]], dtype=np.float64)
    #
    # fp_ref_H1 = 'inputs/H1_ref.txt'
    # if os.path.exists(fp_ref_H1):
    #     ref_H1 = np.loadtxt(fp_ref_H1)
    # else:
    #     ref_H1 = _decompose_to_echelon_form(H)
    #     np.savetxt(fp_ref_H1, ref_H1, delimiter=' ', fmt='%d')
    #
    # fp_hat_H1 = 'inputs/H1_hat.txt'
    # if os.path.exists(fp_hat_H1):
    #     hat_H1 = np.loadtxt(fp_hat_H1)
    # else:
    #     hat_H1 = _decompose_to_echelon_form(ref_H1)
    #     np.savetxt(fp_hat_H1, hat_H1, delimiter=' ', fmt='%d')

    # y = np.loadtxt('inputs/y1.txt')
    # decoded_word, ret_code = decode_vector(hat_H=hat_H, y=None, p=0.1, max_iterations=20)
    # H = np.loadtxt('inputs/H1.txt')/

