import numpy as np

"""
 
"""


def _is_full_rank(mat):
    rank = np.linalg.matrix_rank(mat)
    is_full_rank = rank == min(mat.shape)
    return is_full_rank


def _rearrange_to_systematic_form(ref):
    """
    Rearrange given row-echelon form to systematic form, whereby one side has an identity matrix of shape (n-k,
    n-k) separated out from the rest of the matrix, termed `P` (in LDPC slides).
    :param ref: Row-echelon form of parity check matrix. As 2d array, same shape as parity check matrix, (n-k, n).
    :return: Systematic form of parity check matrix. As 2d array, same shape as parity check matrix, (n-k, n).
    """
    hat_H = ref.copy()
    n_minus_k, n = hat_H.shape
    # identity matrix we want on left side
    id_mat = np.eye(n_minus_k)
    for i in range(n_minus_k):
        for j in range(i, n):
            if np.array_equal(hat_H[:, j], id_mat[:, i]):
                # swap cols
                hat_H[:, [i, j]] = hat_H[:, [j, i]]
                break
    return hat_H


def _make_systematic_encoding_matrix(hat_H, n_minus_k):
    """
    Convert systematic form of parity check matrix `H_hat` into a systematic encoding matrix (`G`) which is a
    column-wise concatenation of a matrix referred to as `P` shape (n-k, k) and an identity matrix of shape (k, k).
    :param hat_H: Systematic form of parity check matrix. A row-wise concatenation of identity matrix (n-k, n-k)
    and "something else" `P` (n-k, k). Shown as [I_n-k | P] in slides.
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
    ref = H.copy()  # in case the parity check matrix is needed in subsequent operations..
    rows, cols = ref.shape
    for i in range(rows):
        if ref[i, i] == 0:
            for j in range(i + 1, rows):
                if ref[j, i] != 0:
                    ref[[i, j]] = ref[[j, i]]  # Swap rows
                    break
        for j in range(i + 1, rows):
            if ref[j, i] == 0:
                continue
            ratio = (ref[j, i] / ref[i, i]) % 2
            scal_by_rt = (ratio * ref[i, i:]) % 2
            ref[j, i:] = (ref[j, i:] - scal_by_rt) % 2
    return ref


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
        return

    ref = _decompose_to_echelon_form(H)
    hat_H = _rearrange_to_systematic_form(ref)
    I_part_of_hat_H = hat_H[:n_minus_k, :n_minus_k]
    I_n_k = np.eye(n_minus_k).astype(int)

    if not np.array_equal(I_part_of_hat_H, I_n_k):
        print(f'Failed to rearrange the ref into a systematic form with identity matrix on left side')
        return None

    G = _make_systematic_encoding_matrix(hat_H, n_minus_k)

    return hat_H, G


if __name__ == '__main__':
    build_systematic_encoding_matrix()
    pass
