from pysph.sph.equation import Equation
from compyle.api import declare
from math import pow, sqrt, pi, acos, atan2, cos, sin, fabs


def vector_normalize(vec=[1.0, 1.0, 1.0], vec_size=3):
    r"""
    Returns a normalized vector, i.e. with unity norm.

    Parameters
    ----------
    :param vec: list representing a vector or vector of vectors
    :param vec_size: integer representing the number elements in the vector

    Output
    -----------
    :return: None
    """
    i, j, num_vecs = declare("int", 3)

    num_vecs = int(vec_size / 3)
    for i in range(num_vecs):
        vec_sum2 = 0.0
        for j in range(3):
            vec_sum2 += pow(vec[3*i + j], 2)
        vec_sum = sqrt(vec_sum2)
        for j in range(3):
            vec[3*i + j] /= vec_sum


def vector_outer_product(vec1=[1.0, 1.0, 1.0], vec2=[1.0, 1.0, 1.0],
                         mat=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                         n=2):
    r"""
    Given 2 vectors (1x3), calculates the outer product, which is a matrix of
    size (3x3).

    Parameters
    ----------
    :param vec1: list representing the first vector
    :param vec2: list representing the second vector
    :param mat: resultant matrix
    :param n: dimension of the vector

    Output
    -----------
    :return: None
    """
    i, j = declare("int", 2)

    for i in range(n):
        for j in range(n):
            mat[n*i + j] = vec1[i]*vec2[j]


def matrix_multiply(mat1=[1.0, 1.0], mat2=[1.0, 1.0], res=[1.0, 1.0], n=2):
    r"""
    Multiply two square matrices. Stores the result in 'res'.

    Parameters
    ----------
    :param mat1: list representing matrix 1
    :param mat2: list representing matrix 2
    :param res: list to hold the result
    :param n: integer representing the number of rows (or columns)

    Output
    -----------
    :return: None
    """
    i, j, k = declare('int', 3)

    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += mat1[n*i + k] * mat2[n*k + j]
            res[n*i + j] = s


def matrix_trace(mat=[1.0, 1.0, 1.0, 1.0], n=2):
    r"""
    Returns the sum of the diagonal elements of a square matrix.

    Parameters
    ----------
    :param mat: list representing a matrix
    :param n: integer representing the number of rows (or columns)

    Output
    -----------
    :return: float corresponding to the matrix's trace
    """
    if n == 2:
        return mat[0] + mat[3]

    else:
        return mat[0] + mat[4] + mat[8]


def matrix_determinant_2x2(mat=[1.0, 1.0, 1.0, 1.0]):
    r"""
    Returns the determinant of a square 2x2 matrix.

    Parameters
    ----------
    :param mat: list representing a matrix

    Output
    -----------
    :return: float corresponding to the matrix's determinant
    """
    return mat[0]*mat[3] - mat[1]*mat[2]


def matrix_determinant(mat=[1.0, 1.0, 1.0, 1.0], n=2):
    r"""
    Returns the determinant of 3x3 matrix.

    Parameters
    ----------
    :param mat: list representing a matrix
    :param n: integer representing the number of rows (or columns)

    Output
    -----------
    :return: float corresponding to the matrix's determinant
    """
    i = declare("int")
    sub_a, sub_b, sub_c = declare("matrix(4)", 3)

    if n == 2:
        return matrix_determinant_2x2(mat)

    else:
        res = 0.0

        # Matrix co-factors
        a_coff, b_coff, c_coff = mat[0:3]

        # Extract sub-matrices
        for i in range(2):
            sub_a[i] = mat[i+4]
            sub_a[i+2] = mat[i+7]
            sub_b[i] = mat[2*i + 3]
            sub_b[i+2] = mat[2*i + 6]
            sub_c[i] = mat[i+3]
            sub_c[i+2] = mat[i+6]

        # Determinant calculation
        res += (
            a_coff*matrix_determinant_2x2(sub_a)
            - b_coff*matrix_determinant_2x2(sub_b)
            + c_coff*matrix_determinant_2x2(sub_c)
        )

        return res


def matrix_transpose(mat=[1.0, 1.0, 1.0, 1.0], res=[1.0, 1.0, 1.0, 1.0], n=2):
    r"""
    Transpose a square matrix, by flipping rows by columns of same index.
    Stores the result in 'res'.

    Parameters
    ----------
    :param mat: list representing matrix
    :param res: list to hold the result
    :param n: integer representing the number of rows (or columns)

    Output
    -----------
    :return: None
    """
    i, j, idx, m = declare("int", 4)

    if n == 2:
        res[0] = mat[0]
        res[1] = mat[2]
        res[2] = mat[1]
        res[3] = mat[3]

    else:
        m = n + 1
        for i in range(n):
            for j in range(n):
                idx = n*i + j
                mij = mat[idx]
                if idx % m == 0:
                    res[idx] = mat[idx]
                else:
                    res[idx] = mat[n*j + i]


def matrix_eigenvalues(mat=[1.0, 1.0, 1.0, 1.0], eigvals=[1.0, 1.0], n=2):
    r"""
    Algorithm based on the closed form solution of eigenvalues of a symmetric
    3x3 matrix proposed in:

     Smith, O.K. (1961) Eigenvalues of a symmetric 3x3 matrix. Communications
     of the ACM 4(4), p168.

     (https://dl.acm.org/citation.cfm?doid=355578.366316 - Accessed 02/21/2019)

     For an explanation on the meaning of each term in the code below, refer to
     the paper.

    Parameters
    -----------
    :param mat: list representing the matrix in Voight notation
    :param eigvals: list with eigenvalues sorted in ascending order
    :param n: int representing the number or rows (columns) in 'mat'

    Output
    -----------
    :return: None
    """
    i = declare("int")
    eigvals_temp = declare("matrix(3)")
    mat_b = declare("matrix(9)")

    if n == 2:
        mat_tr = matrix_trace(mat, 2)  # Trace of the matrix
        s4ac = sqrt(pow(mat_tr, 2.0) - 4*matrix_determinant(mat, 2))
        eigvals[0] = (mat_tr - s4ac) / 2.0
        eigvals[1] = (mat_tr + s4ac) / 2.0
        eigvals[2] = 0.0

    else:
        m = matrix_trace(mat, 3) / 3.0
        for i in range(9):
            mat_b[i] = mat[i]
            if i % 4 == 0:
                mat_b[i] -= m

        q = matrix_determinant(mat_b, 3) / 2.0

        p = 0.0
        for i in range(9):
            p += pow(mat_b[i], 2.0) / 6.0

        c = pow(p, 3.0) - pow(q, 2.0)
        if c < 0.0:
            c = 0.0
        phi = atan2(sqrt(c), q) / 3.0

        eigvals_temp[0] = m + 2.0*sqrt(p) * cos(phi)
        eigvals_temp[1] = m - sqrt(p)*(cos(phi) + sqrt(3.0)*sin(phi))
        eigvals_temp[2] = m - sqrt(p) * (cos(phi) - sqrt(3.0) * sin(phi))

        # Sorting the eigenvalues
        if eigvals_temp[0] < eigvals_temp[1]:
            if eigvals_temp[1] < eigvals_temp[2]:
                eigvals[0] = eigvals_temp[0]
                eigvals[1] = eigvals_temp[1]
                eigvals[2] = eigvals_temp[2]
            elif eigvals_temp[0] < eigvals_temp[2]:
                eigvals[0] = eigvals_temp[0]
                eigvals[1] = eigvals_temp[2]
                eigvals[2] = eigvals_temp[1]
            else:
                eigvals[0] = eigvals_temp[2]
                eigvals[1] = eigvals_temp[0]
                eigvals[2] = eigvals_temp[1]
        else:
            if eigvals_temp[2] < eigvals_temp[1]:
                eigvals[0] = eigvals_temp[2]
                eigvals[1] = eigvals_temp[1]
                eigvals[2] = eigvals_temp[0]
            elif eigvals_temp[0] < eigvals_temp[2]:
                eigvals[0] = eigvals_temp[1]
                eigvals[1] = eigvals_temp[0]
                eigvals[2] = eigvals_temp[2]
            else:
                eigvals[0] = eigvals_temp[1]
                eigvals[1] = eigvals_temp[2]
                eigvals[2] = eigvals_temp[0]


def matrix_eigenvectors(mat=[1.0, 1.0, 1.0, 1.0], eigvals=[1.0, 1.0],
                        eigvecs=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], n=2):
    r"""
    Algorithms taken from:
        http://www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/
    as of 03/07/2019.

    and

        https://en.wikipedia.org/wiki/Eigenvalue_algorithm as of 02/20/2019.

    Parameters
    -----------
    :param mat: list representing the matrix in Voight notation
    :param eigvals: list with eigenvalues sorted in ascending order
    :param eigvecs: list holding the resulting eigenvectors for 'mat'
    :param n: int representing the number or rows (columns) in 'mat'

    Output
    -----------
    :return: None
    """
    tol = 1e-12
    i, j, k, max_index = declare("int", 4)
    col_sum = declare("matrix(3)")
    temp_a, temp_b, temp_c = declare("matrix(9)", 3)

    if n == 2:
        # Check for diagonal matrices.
        diag_sum = fabs(mat[1]) + fabs(mat[2])
        if diag_sum < tol:
            eigvecs[0] = 1.0
            eigvecs[2] = 1.0

        else:
            if mat[2] != 0:
                eigvecs[0] = eigvals[0] - mat[3]
                eigvecs[1] = mat[2]
                eigvecs[3] = eigvals[1] - mat[3]
                eigvecs[4] = mat[2]
            elif mat[1] != 0:
                eigvecs[0] = mat[1]
                eigvecs[1] = eigvals[0] - mat[0]
                eigvecs[3] = mat[1]
                eigvecs[4] = eigvals[1] - mat[0]
            else:
                eigvecs[0] = 0.0
                eigvecs[1] = 1.0
                eigvecs[3] = 1.0
                eigvecs[4] = 0.0

            eigvecs[2] = 0.0
            eigvecs[5] = 0.0
            vector_normalize(eigvecs, 3 * n)

    else:

        # Check for diagonal matrices.
        diag_sum = 0.0
        for i in range(9):
            if i % 4 != 0.0:
                diag_sum += fabs(mat[i])
        if diag_sum < tol:
            eigvecs[0] = 1.0
            eigvecs[4] = 1.0
            eigvecs[8] = 1.0

        else:
            for i in range(3):
                max_value = -1.0e32

                for j in range(9):
                    temp_a[j] = mat[j]
                    temp_b[j] = mat[j]

                if i == 0:
                    for j in range(0, 9, 4):
                        temp_a[j] -= eigvals[1]
                        temp_b[j] -= eigvals[2]

                    matrix_multiply(temp_a, temp_b, temp_c, 3)

                    # Find column with maximum sum
                    for j in range(3):
                        col_sum[j] = 0.0
                        for k in range(3):
                            col_sum[j] += abs(temp_c[3*k + j])
                    for j in range(3):
                        if col_sum[j] > max_value:
                            max_index = j
                            max_value = col_sum[j]

                    # Assign values to the vector list
                    for k in range(3):
                        eigvecs[k] = temp_c[3*k + max_index]

                elif i == 1:
                    for j in range(0, 9, 4):
                        temp_a[j] -= eigvals[0]
                        temp_b[j] -= eigvals[2]
                    matrix_multiply(temp_a, temp_b, temp_c, 3)

                    for j in range(3):
                        col_sum[j] = 0.0
                        for k in range(3):
                            col_sum[j] += abs(temp_c[3*k + j])
                    for j in range(3):
                        if col_sum[j] > max_value:
                            max_index = j
                            max_value = col_sum[j]

                    for k in range(3):
                        eigvecs[k+3] = temp_c[3*k + max_index]

                else:
                    for j in range(0, 9, 4):
                        temp_a[j] -= eigvals[0]
                        temp_b[j] -= eigvals[1]
                    matrix_multiply(temp_a, temp_b, temp_c, 3)

                    for j in range(3):
                        col_sum[j] = 0.0
                        for k in range(3):
                            col_sum[j] += abs(temp_c[3*k + j])
                    for j in range(3):
                        if col_sum[j] > max_value:
                            max_index = j
                            max_value = col_sum[j]

                    for k in range(3):
                        eigvecs[k+6] = temp_c[3*k + max_index]

            vector_normalize(eigvecs, 3*n)


def tensor_multiply_scalar(mat=[1.0, 1.0, 1.0, 1.0], a=0.0,
                           res=[1.0, 1.0, 1.0, 1.0], n=2):
    r"""
    This equation multiplies a matrix (mat) by a scalar value (a) and returns
    the result in a second matrix (res).

    :param mat:
    :param a:
    :param res:
    :param n:
    :return:
    """
    i = declare("int")
    for i in range(n):
        res[i] = a*mat[i]


def matrix_add(mat1=[1.0, 1.0, 1.0, 1.0], mat2=[1.0, 1.0, 1.0, 1.0],
               res=[1.0, 1.0, 1.0, 1.0], n=2, s=1.0):
    r"""
    This equation adds two matrices of equal dimension (mat1, mat2) and returns
    the result in a second matrix (res).

    :param mat1:
    :param mat2:
    :param res:
    :param s: -1.0 if subtraction
    :param n:
    :return: None
    """
    i = declare("int")
    for i in range(n):
        res[i] = mat1[i] + s*mat2[i]


def matrix_multiply_vector(mat=[1.0, 1.0, 1.0, 1.0], vec=[0.0, 0.0],
                           res=[0.0, 0.0], n=2):
    r"""
    This equation multiplies a square matrix (mat) by a vector (vec) and
    returns the result in a second vector (res).

    :param mat:
    :param vec:
    :param res:
    :param n:
    :return:
    """
    i, j = declare("int", 2)

    # Initialize result vector
    for i in range(n):
        res[i] = 0.0

    for i in range(n):
        for j in range(n):
            res[i] += mat[n*i + j]*vec[j]


def matrix_exponentiation(mat=[0.0, 0.0], res=[0.0, 0.0], exp=0, n=2):
    r"""
    This function returns the input matrix (mat) to the power of (exp), i.e.
    it multiplies the matrix by itself exp-times.

    :param mat: List (double)
    :param res: List (double)
    :param exp: (int)
    :param n: (int)
    :return: None
    """
    i, j, k, m = declare("int", 4)
    res_old = declare("matrix(36)")

    for i in range(n*n):
        res[i] = 0.0

    m = n + 1
    if exp == 0:
        for i in range(n*n):
            if i % m == 0:
                res[i] = 1.0

    else:
        matrix_exponentiation(mat, res, exp-1, n)

        for i in range(n*n):
            res_old[i] = res[i]

        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += mat[n*i + k]*res_old[n*k + j]
                res[n*i + j] = s


def augment_voigt_tensor(mat=[0.0, 0.0, 0.0], res=[0.0, 0.0, 0.0], n=2):
    r"""

    :param mat:
    :param res:
    :param n:
    :return:
    """
    i = declare("int")

    for i in range(2*n):
        m = mat[i]
        if i < n:
            res[4*i] = m
        elif i < 2*n-1:
            res[i-2] = m
            res[(i-2)*n] = m
        else:
            res[i] = m
            res[i+2] = m


def matrix_inverse(mat=[0.0, 0.0, 0.0, 0.0], res=[0.0, 0.0, 0.0, 0.0], n=2):
    r"""
    This function takes a square, symmetric, positive-definite matrix (mat) and
    returns its inverse. It first performs a Cholesky decomposition to
    make the matrix lower triangular and then invert it.

    References
    -----------
    https://en.wikipedia.org/wiki/Cholesky_decomposition

    https://math.stackexchange.com/questions/1003801/...
        ...inverse-of-an-invertible-upper-triangular-matrix-of-order-3

    (As of 04/7/2020)

    :param mat:
    :param res:
    :param n:
    :return: None
    """
    i, j, k, idx, m, r = declare("int", 6)
    low_mat, low_inv, temp, diag_inv, band, mat_t, d = declare("matrix(36)", 7)
    ten = declare("matrix(15)")
    matv, temp2 = declare("matrix(36)", 2)

    # TODO: First test if matrix is positive-definite!

    # This is necessary to convert a tensor in Voigt notation to a full square
    #  tensor
    if n % 2 > 0:
        r = int((n*n + n)/2)
        for i in range(r):
            ten[i] = mat[i]
        augment_voigt_tensor(ten, matv, n)
    else:
        for i in range(n*n):
            matv[i] = mat[i]

    # Initialize lower triangular matrix, low_mat
    m = int(n + 1)
    for i in range(n*n):
        low_mat[i] = 0.0
        diag_inv[i] = 0.0
        band[i] = 0.0
        temp[i] = 0.0
        temp2[i] = 0.0
        d[i] = 0.0

    # # Perform Cholesky decomposition
    # for i in range(n):
    #     for j in range(n):
    #         idx = n*i + j
    #         lsum = 0.0
    #
    #         # Diagonal terms
    #         if idx % (n+1) == 0:
    #             for k in range(j):
    #                 ljk = low_mat[n*i + k]
    #                 lsum += ljk*ljk
    #             low_mat[idx] = sqrt(matv[idx] - lsum)
    #
    #         # Off-diagonal terms
    #         elif i > j:
    #             for k in range(j):
    #                 lsum += low_mat[n*i + k]*low_mat[n*j + k]
    #             low_mat[idx] = (matv[idx] - lsum) / low_mat[(n+1)*j]

    # Perform LDL decomposition
    for i in range(n*n):
        if i % m == 0:
            low_mat[i] = 1.0

    for i in range(n):
        for j in range(n):
            idx = n*i + j
            lsum = 0.0

            # Diagonal terms
            if idx % m == 0:
                for k in range(j):
                    ljk = low_mat[n*i + k]
                    lsum += ljk*ljk*d[m*k]
                d[idx] = matv[idx] - lsum

            # Off-diagonal terms
            elif i > j:
                for k in range(j):
                    lsum += low_mat[n*i + k]*low_mat[n*j + k]*d[m*k]
                low_mat[idx] = (matv[idx] - lsum) / d[m*j]

    # Perform diagonalizing
    for i in range(n):
        d[i*m] = sqrt(d[i*m])
    matrix_multiply(low_mat, d, temp, n)
    for i in range(n*n):
        low_mat[i] = temp[i]

    # Inversion of the lower diagonal matrix
    for i in range(n*n):
        if i % m == 0:
            diag_inv[i] = -1.0/low_mat[i]  # Matrix with inverse diag terms
        else:
            band[i] = low_mat[i]  # Matrix diagonal terms equal to zero

    matrix_multiply(diag_inv, band, mat_t, n)

    for i in range(n):
        matrix_exponentiation(mat_t, temp, i, n)
        for j in range(n*n):
            temp2[j] += temp[j]

    matrix_multiply(temp2, diag_inv, low_inv, n)
    matrix_transpose(low_inv, temp, n)
    matrix_multiply(temp, low_inv, temp2, n)

    # Restore vector back to Voigt notation
    if n % 2 > 0:
        tensor_voight(temp2, res, n)


def tensor_voight(A=[0.0, 0.0, 0.0, 0.0], res=[0.0, 0.0, 0.0], n=2):
    r"""
    This function transforms a full nxn square symmetric tensor in its Voight
    form, i.e., a vector of dimensions 1x(n^2 + n)/2.

    :param A:
    :param res:
    :param n:
    :return:
    """
    i, j, m, idx, idx2 = declare("int", 3)

    m = n + 1
    idx = 0
    idx2 = 0
    for i in range(n*n):
        if i % m == 0:
            res[idx] = A[i]
            idx += 1
        elif i < idx*n:
            res[n + idx2] = A[i]
            idx2 += 1


def matrix_contract(mat1=[0.0, 0.0], mat2=[0.0, 0.0], res=[0.0], n=2):
    r"""
    This function takes 2 square matrices representing a n-order tensor and
    contract them to the n-order, i.e., returning a scalar. This scalar is the
    sum of the element-wise product between the two matrices

    :param mat1:
    :param mat2:
    :param res:
    :param n:
    :return: None
    """
    i = declare("int")

    for i in range(n):
        res[0] += mat1[i]*mat2[i]


def tensor_voight_multiply(vec1=[0.0, 0.0], vec2=[0.0, 0.0], res=[0.0, 0.0]):
    r"""
    This function calculates the dot product (or matrix multiplication) of two
    tensors in Voight notation.

    :param vec1:
    :param vec2:
    :param res:
    :return:
    """
    v1xx = vec1[0]
    v1yy = vec1[1]
    v1zz = vec1[2]
    v1xy = vec1[3]
    v1xz = vec1[4]
    v1yz = vec1[5]
    v2xx = vec2[0]
    v2yy = vec2[1]
    v2zz = vec2[2]
    v2xy = vec2[3]
    v2xz = vec2[4]
    v2yz = vec2[5]
    res[0] = v1xx*v2xx + v1xy*v2xy + v1xz*v2xz
    res[1] = v1yy*v2yy + v1xy*v2xy + v1yz*v2yz
    res[2] = v1zz*v2zz + v1yz*v2yz + v1xz*v2xz
    res[3] = v1xx*v2xy + v1xy*v2yy + v1xz*v2yz
    res[4] = v1xx*v2xz + v1xy*v2yz + v1xz*v2zz
    res[5] = v1xy*v2xz + v1yy*v2yz + v1yz*v2zz


def tensor_voigt_contract(mat1=[0.0, 0.0], mat2=[0.0, 0.0], res=[0.0], n=2):
    r"""
    This function takes 2 square matrices representing a n-order tensor and
    contract them to the n-order, i.e., returning a scalar. This scalar is the
    sum of the element-wise product between the two matrices

    :param mat1:
    :param mat2:
    :param res:
    :param n:
    :return: None
    """
    i, r = declare("int", 2)
    r = int((n*n + n)/2)

    for i in range(r):
        mm = mat1[i]*mat2[i]
        if i < n:
            res[0] += mm
        else:
            res[0] += 2*mm
