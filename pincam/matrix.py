import numpy as np


def matmul_xforms(xforms):
    """Multiply multiply transfromation matrices together.

    This method multiplies matrices in correct post-order multiplication.
    That is matrices are multiplied from left to right.

    For example, for a typical rigid transformation where it is desired to
    multiply rotation around Z axis, rotation around X, and translation, the
    method does the following:

    xforms: [RZ1, RX2, T1]
    P = T1(RX2(RZ1))

    Args:
        list_of_matrices: list of matrices in sequential order.

    Returns:
        Single matrix.
    """
    m = xforms[0]
    for xf in xforms[1:]:
        m = np.matmul(xf, m)
    return m


def xform_translation_matrix(move_vector):
    """
    Modified from Christopher Gohlke
    https://www.lfd.uci.edu/~gohlke/code/transformations.py.html

    :param move_vector: 3 x 1 direction vector
    :return: 4 x 4 translation matrix
    [[1, 0, 0, move_vector[0]],
    [0, 1, 0, move_vector[1]],
    [0, 0, 1, move_vector[2]],
    [0, 0, 0, 1.0]]
    """
    m = np.identity(4)
    m[:3, 3] = move_vector[:3]
    return m


def xform_rotation_matrix(vector_origin, vector_axis, theta):
    """
    Modified from Christopher Gohlke
    https://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    If this doesn't work, go back to Rodriguez rotation method in matrixutils.
    """

    sina = np.sin(theta)
    cosa = np.cos(theta)
    vector_axis = vector_axis / np.linalg.norm(vector_axis)

    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(vector_axis, vector_axis) * (1.0 - cosa)
    vector_axis *= sina
    R += np.array([[0.0, -vector_axis[2], vector_axis[1]],
                   [vector_axis[2], 0.0, -vector_axis[0]],
                   [-vector_axis[1], vector_axis[0], 0.0]])
    M = np.identity(4)
    M[:3, :3] = R

    if vector_origin is not None:
        # rotation not around origin
        M[:3, 3] = vector_origin - np.dot(R, vector_origin)

    return M