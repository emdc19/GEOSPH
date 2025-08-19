from pysph.sph.equation import Equation
from compyle.api import declare
from math import fabs, sqrt, log, pow, exp, tan, isnan
import sys

from matrix_operations import matrix_multiply, matrix_add, matrix_transpose, \
    tensor_multiply_scalar, matrix_multiply_vector, matrix_inverse, \
    matrix_exponentiation, tensor_voight, tensor_voight_multiply, \
    matrix_contract, vector_outer_product, augment_voigt_tensor, \
    tensor_voigt_contract, matrix_determinant

"""
============================== Helper Functions ===============================
"""


def yield_criterion(q=[0.0], p=[0.0], aphi=0.0, ac=0.0, sy=0.0, y_criterion=1):
    """
    Calculates the yield function value.

    If the value calculated is greater than zero, it means the material is
     yielding at the particle. It only implements one type of yield function
     called generalized Von Mises. If aphi is zero, returns the Von Mises yield
     criterion, otherwise, returns the Drucker-Prager criterion.

    Parameters
    ----------
    :param p:
    :param q:
    :param aphi: -- alpha_phi
    :param ac: -- alpha_c
    :param sy: -- kappa
    :param y_criterion:

    Output
    -----------
    :return:
    """
    if y_criterion:
        return sqrt(2.0/3.0)*q[0] + aphi*p[0] - ac*sy
    else:
        return -1

def yield_criterion2(q=0.0, p=0.0, aphi=0.0, ac=0.0, sy=0.0):
    """
    Calculates the yield function value.

    If the value calculated is greater than zero, it means the material is
     yielding at the particle. It only implements one type of yield function
     called generalized Von Mises. If aphi is zero, returns the Von Mises yield
     criterion, otherwise, returns the Drucker-Prager criterion.

    Parameters
    ----------
    :param p:
    :param q:
    :param aphi: -- alpha_phi
    :param ac: -- alpha_c
    :param sy: -- kappa

    Output
    -----------
    :return:
    """
    
    return sqrt(2.0/3.0)*q + aphi*p - ac*sy


def hardening_modulus(acc_eps_p=0.0, hard_mod0=0.0):
    # Implement non-linear functions of the accumulated plastic strain for
    #  non-linear hardening modulus.
    return hard_mod0


def yield_stress(acc_eps_p=0.0, hard_mod=0.0, sy0=0.0):
    r"""
    This function returns the updated yield stress based on the accumulated
    plastic strain and hardening modulus

    :param acc_eps_p:
    :param hard_mod:
    :param sy0:
    :return:
    """
    return sy0 + hard_mod*acc_eps_p


def yield_criterion_grad(q=[0.0], sig_dev_v=[0.0, 0.0], aphi=0.0, ac=0.0,
                         dfds=[0.0, 0.0], dfdk=[0.0]):
    i = declare("int")
    n = declare("matrix(6)")  # Unit deviatoric stress tensor

    norm_s = sqrt(2.0/3.0)*q[0]
    for i in range(6):
        p_i = 0.0
        n_i = sig_dev_v[i] / norm_s
        n[i] = n_i
        if i < 3:
            p_i = aphi / 3.0
        dfds[i] = n_i + p_i

    dfdk[0] = -ac

    printf("\n")
    printf("q_tr")
    printf("%.6f \n", q[0])
    printf("\n")
    printf("ac")
    printf("%.6f \n", ac)
    printf("\n")
    printf("First derivative of the yield function: dF/dSig")
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", dfds[0], dfds[1], dfds[2],
           dfds[3], dfds[4], dfds[5])
    printf("\n")
    printf("dF/dK")
    printf("%.6f \n", dfdk[0])
    printf("\n")


def flow_potential_derivatives(q=[0.0], sig_dev_v=[0.0, 0.0], apsi=0.0, ac=0.0,
                               dqds=[0.0, 0.0], dqdk=[0.0], d2qds2=[0.0, 0.0],
                               d2qdk2=[0.0], d2qdsdk=[0.0, 0.0],
                               eye_dev=[0.0, 0.0]):
    r"""
    This function calculates the derivatives of the plastic flow potential with
    respect to the stress tensor Sigma and the yield stress Sigma_y

    :param q:
    :param sig_dev_v:
    :param apsi:
    :param ac:
    :param dqds:
    :param dqdk:
    :param d2qds2:
    :param d2qdk2:
    :param d2qdsdk:
    :param eye_dev:
    :return:
    """
    i, j = declare("int", 2)
    n = declare("matrix(6)")  # Unit deviatoric stress tensor

    # First derivative of the plastic potential Q with respect to Sigma
    norm_s = sqrt(2.0/3.0)*q[0]
    for i in range(6):
        p_i = 0.0
        n_i = sig_dev_v[i] / norm_s
        n[i] = n_i
        if i < 3:
            p_i = apsi / 3.0
        dqds[i] = n_i + p_i

    for i in range(6):
        d2qdsdk[i] = 0.0

    # First derivative of Q with respect to the yield stress
    dqdk[0] = -ac

    # Double derivative of Q with respect to Sigma
    for i in range(6):
        n_i = n[i]
        for j in range(6):
            d2qds2[6*i + j] = (eye_dev[6*i + j] - n_i*n[j]) / norm_s

    # Double derivative of Q with respect to the yield stress
    d2qdk2[0] = 0.0

    printf("\n")
    printf("Deviatoric unit tensor n: S/|S|")
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", n[0], n[1], n[2], n[3], n[4],
           n[5])
    printf("\n")
    printf("First derivative of the plastic potential: dQ/dSig")
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", dqds[0], dqds[1], dqds[2],
           dqds[3], dqds[4], dqds[5])
    printf("\n")
    printf("dQ/dK")
    printf("%.6f \n", dqdk[0])
    printf("\n")
    printf("d2Q/dK2")
    printf("%.6f \n", d2qdk2[0])
    printf("\n")
    printf("Second derivative of the plastic potential: d2Q/dSig2")
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", d2qds2[0], d2qds2[1], d2qds2[2],
           d2qds2[3], d2qds2[4], d2qds2[5])
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", d2qds2[6], d2qds2[7], d2qds2[8],
           d2qds2[9], d2qds2[10], d2qds2[11])
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", d2qds2[12], d2qds2[13],
           d2qds2[14], d2qds2[15], d2qds2[16], d2qds2[17])
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", d2qds2[18], d2qds2[19],
           d2qds2[20], d2qds2[21], d2qds2[22], d2qds2[23])
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", d2qds2[24], d2qds2[25],
           d2qds2[26], d2qds2[27], d2qds2[28], d2qds2[29])
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", d2qds2[30], d2qds2[31],
           d2qds2[32], d2qds2[33], d2qds2[34], d2qds2[35])
    printf("\n")
    printf("Second derivative of the plastic potential: d2Q/dSigdK")
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", d2qdsdk[0], d2qdsdk[1],
           d2qdsdk[2], d2qdsdk[3], d2qdsdk[4], d2qdsdk[5])
    printf("\n")


def residuals(de=[0.0, 0.0], dsigma=[0.0, 0.0], dqds=[0.0, 0.0], dqdk=[0.0],
              dk=0.0, dg=0.0, hard_mod=0.0, r1=[0.0, 0.0], r2=[0.0]):
    r"""
    This function evaluates the residuals defined in the return mapping
    algorithm.

    :param de:
    :param dsigma:
    :param dqds:
    :param dqdk:
    :param dk:
    :param dg:
    :param hard_mod:
    :param r1:
    :param r2:
    :return: None
    """
    i = declare("int")
    res = declare("matrix(6)")

    matrix_multiply_vector(de, dsigma, res, 6)
    for i in range(6):
        r1[i] = -res[i] + dg*dqds[i]

    r2[0] = dk/hard_mod + dg*dqdk[0]

    printf("\n")
    printf("Residual - R1")
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", r1[0], r1[1], r1[2], r1[3],
           r1[4], r1[5])
    printf("\n")
    printf("Residual - R2")
    printf("%.6f\n", r2[0])
    printf("\n")


def residuals_hessian(de=[0.0, 0.0], d2qds2=[0.0, 0.0], d2qdsdk=[0.0, 0.0],
                      d2qdk2=[0.0], hard_mod=0.0, dg=0.0, D=[0.0, 0.0],
                      B=[0.0, 0.0], L=[0.0]):
    r"""
    This function returns the Hessian of the residuals used in the general
    return mapping algorithm for isotropic hardening.

    :param de:
    :param d2qds2:
    :param d2qdsdk:
    :param d2qdk2:
    :param hard_mod:
    :param dg:
    :param D:
    :param B: (double - 1x9)
    :param L:
    :return:
    """
    i = declare("int")

    h1 = 0.0
    if hard_mod != 0.0:
        h1 = 1/hard_mod

    for i in range(6):
        D[i] = de[i] + dg*d2qds2[i]
        B[i] = dg*d2qdsdk[i]

    for i in range(6, 36):
        D[i] = de[i] + dg*d2qds2[i]

    L[0] = h1 + dg*d2qdk2[0]  # So that it overloads the input value

    printf("\n")
    printf("Hardening modulus")
    printf("%.3f\n", hard_mod)
    printf("\n")
    printf("Hessian term - D")
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", D[0], D[1], D[2], D[3],
           D[4], D[5])
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", D[6], D[7], D[8], D[9],
           D[10], D[11])
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", D[12], D[13], D[14], D[15],
           D[16], D[17])
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", D[18], D[19], D[20], D[21],
           D[22], D[23])
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", D[24], D[25], D[26], D[27],
           D[28], D[29])
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", D[30], D[31], D[32], D[33],
           D[34], D[35])
    printf("\n")
    printf("Hessian term - B")
    printf("%.6f %.6f %.6f %.6f %.6f %.6f\n", B[0], B[1], B[2], B[3], B[4],
           B[5])
    printf("\n")
    printf("Hessian term - G")
    printf("%.6f\n", L[0])
    printf("\n")


def elasticity_tensors(ce=[0.0, 0.0], de=[0.0, 0.0], eye_dev=[0.0, 0.0], K=0.0,
                       G=0.0):
    r"""
    This function returns the 4th-order tangent elastic tensors Ce and De=1/Ce
    (Hooke's law)

    :param ce:
    :param de:
    :param eye_dev:
    :param G:
    :param K:
    :return:
    """
    i, j = declare("int", 2)
    proj11 = declare("matrix(18)")

    # Define projection tensor 1x1
    for i in range(3):
        for j in range(6):
            id_ij = 0.0
            if j < 3:
                id_ij = 1.0
            proj11[6*i + j] = id_ij

    for i in range(6):
        for j in range(6):
            eye_ij = eye_dev[6*i + j]
            id_ij = proj11[6*i + j]
            if i < 3:
                ce[6*i + j] = K*id_ij + 2*G*eye_ij
                de[6*i + j] = id_ij/(9*K) + eye_ij/(2*G)
            else:
                ce[6*i + j] = 2*G*eye_ij
                de[6*i + j] = eye_ij/(2*G)

    printf("\n")
    printf("Elasticity tensor - Ce")
    printf("%.3f %.3f %.3f %.3f %.3f %.3f\n", ce[0], ce[1], ce[2], ce[3],
           ce[4], ce[5])
    printf("%.3f %.3f %.3f %.3f %.3f %.3f\n", ce[6], ce[7], ce[8], ce[9],
           ce[10], ce[11])
    printf("%.3f %.3f %.3f %.3f %.3f %.3f\n", ce[12], ce[13], ce[14], ce[15],
           ce[16], ce[17])
    printf("%.3f %.3f %.3f %.3f %.3f %.3f\n", ce[18], ce[19], ce[20], ce[21],
           ce[22], ce[23])
    printf("%.3f %.3f %.3f %.3f %.3f %.3f\n", ce[24], ce[25], ce[26], ce[27],
           ce[28], ce[29])
    printf("%.3f %.3f %.3f %.3f %.3f %.3f\n", ce[30], ce[31], ce[32], ce[33],
           ce[34], ce[35])
    printf("\n")
    printf("Compliance tensor - De")
    printf("%.6e %.6e %.6e %.6e %.6e %.6e\n", de[0], de[1], de[2], de[3],
           de[4], de[5])
    printf("%.6e %.6e %.6e %.6e %.6e %.6e\n", de[6], de[7], de[8], de[9],
           de[10], de[11])
    printf("%.6e %.6e %.6e %.6e %.6e %.6e\n", de[12], de[13], de[14], de[15],
           de[16], de[17])
    printf("%.6e %.6e %.6e %.6e %.6e %.6e\n", de[18], de[19], de[20], de[21],
           de[22], de[23])
    printf("%.6e %.6e %.6e %.6e %.6e %.6e\n", de[24], de[25], de[26], de[27],
           de[28], de[29])
    printf("%.6e %.6e %.6e %.6e %.6e %.6e\n", de[30], de[31], de[32], de[33],
           de[34], de[35])
    printf("\n")


def deviatoric_identity(eye_dev=[0.0, 0.0]):
    r"""
    This class returns the deviatoric 4th-order identity tensor, which
    multiplied by any symmetric 2nd-order tensor, returns the following vector:
    {Sxx Syy Szz Sxy Sxz Syz}

    :param eye_dev: Double array with dimensions 1-by-36
    :return: None
    """
    i, j = declare("int", 2)

    for i in range(6):
        for j in range(6):
            eye_ij = 0.0
            if i < 3 and i == j:
                eye_ij = 2.0/3.0
            elif i < 3 and i != j and j < 3:
                eye_ij = -1.0/3.0
            elif i >= 3 and i == j:
                eye_ij = 1.0
            eye_dev[6*i + j] = eye_ij

    printf("\n")
    printf("Deviatoric 4th-Order identity tensor")
    printf("%.3f %.3f %.3f %.3f %.3f %.3f\n", eye_dev[0], eye_dev[1],
           eye_dev[2], eye_dev[3], eye_dev[4], eye_dev[5])
    printf("%.3f %.3f %.3f %.3f %.3f %.3f\n", eye_dev[6], eye_dev[7],
           eye_dev[8], eye_dev[9], eye_dev[10], eye_dev[11])
    printf("%.3f %.3f %.3f %.3f %.3f %.3f\n", eye_dev[12], eye_dev[13],
           eye_dev[14], eye_dev[15], eye_dev[16], eye_dev[17])
    printf("%.3f %.3f %.3f %.3f %.3f %.3f\n", eye_dev[18], eye_dev[19],
           eye_dev[20], eye_dev[21], eye_dev[22], eye_dev[23])
    printf("%.3f %.3f %.3f %.3f %.3f %.3f\n", eye_dev[24], eye_dev[25],
           eye_dev[26], eye_dev[27], eye_dev[28], eye_dev[29])
    printf("%.3f %.3f %.3f %.3f %.3f %.3f\n", eye_dev[30], eye_dev[31],
           eye_dev[32], eye_dev[33], eye_dev[34], eye_dev[35])


def identity_tensor(eye=[0.0, 0.0, 0.0], n=2):
    r"""
    This function returns an identity tensor in Voight notation.

    :param eye:
    :param n: (int) Use n=2 or n=3 always (2D and 3D)
    :return: None
    """
    i = declare("int")

    if n > 3:
        n = 3

    for i in range(2*n):
        e = 0.0
        if i < n and i < 3:
            e = 1.0
        eye[i] = e


def decompose_stress(sigma=[0.0, 0.0], sig_dev=[0.0, 0.0], p=[0.0], q=[0.0]):
    i = declare("int")

    pv = (sigma[0] + sigma[4] + sigma[8]) / 3.0
    p[0] = pv

    s2 = 0.0
    for i in range(9):
        s = sigma[i]
        if i % 4 == 0:
            s -= pv
        s2 += s*s
        sig_dev[i] = s

    q[0] = sqrt(3.0*s2 / 2.0)  # sqrt(3*J_2)


def decompose_stress_voight(sigma=[0.0, 0.0], sigma_dev=[0.0, 0.0], p=[0.0],
                            q=[0.0]):
    i = declare("int")

    pv = (sigma[0] + sigma[1] + sigma[2]) / 3.0
    p[0] = pv

    s2 = 0.0
    for i in range(6):
        s = sigma[i]
        if i < 3:
            s -= pv
        else:
            sigma_dev[i] = s
            s /= sqrt(2.0)
        s2 += s*s

    q[0] = sqrt(3.0*s2 / 2.0)  # sqrt(3*J_2)


def yield_criterion_mcc(q=0.0, p=0.0, pc=0.0, ms=0.0):
    """
    Calculates the yield function value.

    If the value calculated is greater than zero, it means the material is
     yielding at the particle. It only implements one type of yield function
     called Modified Cam-Clay (MCC).

    Parameters
    ----------
    :param p:
    :param q:
    :param pc:
    :param ms:

    Output
    -----------
    :return: y = numerical result of yield function
    """
    return (q * q) / (ms * ms) + p * (p - pc)

def yield_criterion_mcc_tensile(q=0.0, p=0.0, pc=0.0, pt=0.0, ms=0.0):
    """
    Calculates the yield function value.

    If the value calculated is greater than zero, it means the material is
     yielding at the particle. It only implements one type of yield function
     called Modified Cam-Clay (MCC).

    Parameters
    ----------
    :param p:
    :param q:
    :param pc:
    :param ms:

    Output
    -----------
    :return: y = numerical result of yield function
    """
    return (q * q) / (ms * ms) + (p-pt) * (p - pc)

class NonlocalEqStrain(Equation):
    def __init__(self, dest, sources, nu, debug=0):
        self.nu = nu
        self.debug_bound = debug
        super(NonlocalEqStrain, self).__init__(dest, sources)

    def initialize(self, d_idx, d_kd, d_eps, dt, d_eps_dot, d_eps_eq, d_eps_eq_bar):
        
        d_eps_eq_bar[d_idx] = 0.0
        i, idx = declare("int", 2)
        eps = declare('matrix(9)')
        idx = int(9 * d_idx)

        for i in range(9):
            eps[i] = d_eps[idx+i]
            deps = dt*d_eps_dot[idx + i]
            eps[i] += deps

        # Find strain invariants and equivalent strain
        I1= eps[0] + eps[4] + eps[8]

        kd = d_kd[d_idx]
        e2 = 0.0
        for i in range(9):
            e = eps[i]
            if i % 4 == 0:
                e -= I1/3
            e2 += e*e
        q_eps = sqrt(3.0*e2 / 2.0)
        J2=q_eps*q_eps/3

        # J2p = (1/3)*(pow(d_eps[idx],2) + pow(d_eps[idx+4],2) + pow(d_eps[idx+8],2) -\
        #            d_eps[idx] * d_eps[idx + 4] - d_eps[idx + 4] * d_eps[idx + 8] -d_eps[idx] * d_eps[idx + 8]+\
        #             3*(pow(d_eps[idx+1],2)+pow(d_eps[idx+2],2)+pow(d_eps[idx+5],2)))

        eps_eq=I1*(kd-1.0)/(2.0*kd*(1.0-2.0*self.nu))+(1.0/(2.0*kd))*sqrt((pow(kd-1.0,2)/pow(1.0-2.0*self.nu,2)) * pow(I1,2)+(J2*2.0*kd)/pow(1.0+self.nu,2))
        d_eps_eq[d_idx] = eps_eq

        
    def loop(self, d_idx, s_idx, s_m, s_rho, s_eps_eq, d_eps_eq_bar, WIJ):
        m = s_m[s_idx]
        rhoj= s_rho[s_idx]
        eps_eq = s_eps_eq[s_idx]

        eps_eq_bar=(m/rhoj)*WIJ*eps_eq

        # output eqs_eq
        d_eps_eq_bar[d_idx] += eps_eq_bar

        if isnan(d_eps_eq_bar[d_idx]):
            printf("\n")
            printf("mj=%.9f rhoj=%.9f WIJ=%.9f eps_eq=%.9f\n", m, rhoj,
                   WIJ,eps_eq)
            printf("\n")


class GradientEqStrain(Equation):
    def __init__(self, dest, sources, dp, nu, debug=0):
        self.dp = dp
        self.nu = nu
        self.debug_bound = debug
        super(GradientEqStrain, self).__init__(dest, sources)

    def initialize(self, d_idx, d_kd, d_eps, dt, d_eps_dot, d_eps_eq, d_eps_eq_bar):
        
        d_eps_eq_bar[d_idx] = 0.0
        i, idx = declare("int", 2)
        eps = declare('matrix(9)')
        idx = int(9 * d_idx)

        for i in range(9):
            eps[i] = d_eps[idx+i]
            deps = dt*d_eps_dot[idx + i]
            eps[i] += deps

        # Find strain invariants and equivalent strain
        I1= eps[0] + eps[4] + eps[8]

        kd = d_kd[d_idx]
        e2 = 0.0
        for i in range(9):
            e = eps[i]
            if i % 4 == 0:
                e -= I1/3
            e2 += e*e
        q_eps = sqrt(3.0*e2 / 2.0)
        J2=q_eps*q_eps/3

        # J2p = (1/3)*(pow(d_eps[idx],2) + pow(d_eps[idx+4],2) + pow(d_eps[idx+8],2) -\
        #            d_eps[idx] * d_eps[idx + 4] - d_eps[idx + 4] * d_eps[idx + 8] -d_eps[idx] * d_eps[idx + 8]+\
        #             3*(pow(d_eps[idx+1],2)+pow(d_eps[idx+2],2)+pow(d_eps[idx+5],2)))

        eps_eq=I1*(kd-1.0)/(2.0*kd*(1.0-2.0*self.nu))+(1.0/(2.0*kd))*sqrt((pow(kd-1.0,2)/pow(1.0-2.0*self.nu,2)) * pow(I1,2)+(J2*2.0*kd)/pow(1.0+self.nu,2))
        d_eps_eq[d_idx] = eps_eq

        
    def loop(self, d_idx, s_idx, s_m, s_rho, s_eps_eq, d_eps_eq, d_eps_eq_bar, XIJ, R2IJ, EPS, DWIJ):
        l=self.dp*1
        i, idx = declare("int", 2)
        m = s_m[s_idx]
        rhoj= s_rho[s_idx]
        eps_eq_bar = 0.0
        for i in range(3):
            eps_eq_bar += 2.0*pow(l,2)*(m/rhoj)*DWIJ[i]*XIJ[i]*(d_eps_eq[d_idx]-s_eps_eq[s_idx])/(R2IJ + EPS)
            if isnan(eps_eq_bar):
                printf("error")
                # printf("%.9f",s_eps_eq[s_idx])
            
        if isnan(eps_eq_bar):
            printf("\n")
            printf("eps_eq_bar is NaN!")
            printf("\n")
            printf("mj=%.9f rhoj=%.9f\n DWIJ0=%.9f DWIJ1=%.9f DWIJ2=%.9f\n XIJ0=%.9f XIJ1=%.9f XIJ2=%.9f\n R2IJ=%.9f\n", m, rhoj,
                   DWIJ[0],DWIJ[1],DWIJ[2],XIJ[0],XIJ[1],XIJ[2],R2IJ)
            printf("\n")

        # output eqs_eq
        d_eps_eq_bar[d_idx] = eps_eq_bar

        

    def post_loop(self, d_idx, d_eps_eq, d_eps_eq_bar):
        d_eps_eq_bar[d_idx] += d_eps_eq[d_idx]


class DamageModel(Equation):
    def __init__(self, dest, sources, nu=0.3, c_model=1, y_criterion=4, debug=0):
        self.nu = nu
        self.c_model = c_model
        self.y_criterion = y_criterion
        self.debug = debug
        super(DamageModel, self).__init__(dest, sources)

    def initialize(self, d_idx, d_sigma, d_sigma_tr,
                   d_sigma_dot, d_eps_dot, d_eps, d_kd,d_kappa_o, 
                   d_betad, d_alphad, d_dam, d_flag, d_gid, dt):

        i, idx, count = declare("int", 3)
        sig_dot = declare("matrix(9)", 1)

        # Index used to access values in arrays
        idx = int(9 * d_idx)

        # Flag particle to use trial values
        d_flag[d_idx] = 0

        # Initialize stress and stress rate tensors
        for i in range(9):
            sig_dot[i] = d_sigma_dot[idx + i]

        # Find strain invariants and equivalent strain
        I1= d_eps[idx] + d_eps[idx + 4] + d_eps[idx + 8]

        kd = d_kd[d_idx]
        kappa_o = d_kappa_o[d_idx]
        betad = d_betad[d_idx]
        alphad = d_alphad[d_idx]

        e2 = 0.0
        for i in range(9):
            e = d_eps[idx + i]
            if i % 4 == 0:
                e -= I1/3
            e2 += e*e
        q_eps = sqrt(3.0*e2 / 2.0)
        J2=q_eps*q_eps/3

        eps_eq=I1*(kd-1)/((2*kd)*(1-2*self.nu))+(1/(2*kd))*sqrt((pow(kd-1,2)/pow(1-2*self.nu,2))*pow(I1,2)+(J2*2*kd)/pow(1+self.nu,2))
        
        # KKT conditions to find kappa
        if eps_eq > kappa_o:
            kappa=eps_eq
        else:
            kappa=kappa_o

        # Calculate the Damage variable
        deltakappa=kappa-kappa_o
        dam=1-(kappa_o/kappa)*((1-alphad)+alphad*exp(-betad*deltakappa))
        d_dam[d_idx]=dam
        
        # Calculate the final stress rate post damage. Final stress integrated in integrator_steppers.py
        for i in range(9):
            d_sigma_dot[idx + i]=dam*sig_dot[i]


"""
Void ratio is not updated in the MCC used by Sabrina's paper
"""
class NonlocalDamageMCC(Equation):
    def __init__(self, dest, sources, nu, c_model, tol, max_iter, debug):
        self.nu = nu
        self.c_model = c_model
        self.tol = tol
        self.max_iter = max_iter
        self.debug = debug
        super(NonlocalDamageMCC, self).__init__(dest, sources)

    def initialize(self, d_idx, d_sigma, d_sigma_tr, d_eps_eq_bar, d_model,
                   d_p, d_q, d_sigma_dev,d_void_ratio,d_void_ref, d_lambda_d,
                   d_sigma_dot, d_eps_dot, d_spin_dot, d_eps_p_dot, d_ep_acc, d_pc,
                   d_lambda_mcc, d_kappa_mcc, d_ms, d_bulk, d_shear, d_kappa, d_kappa_o, 
                   d_betad, d_alphad, d_dam, d_flag, d_gid, d_eps_p_norm, dt):
        
        i, idx, count = declare("int", 3)
        dam, kappa, eps_eq_bar = declare('float',3)
        sigma, sig_dot, sig_spin, sig_spin_t, spin_dot = declare('matrix(9)', 5)
        
        # Index used to access values in arrays
        idx = int(9 * d_idx)

        eps_eq_bar = d_eps_eq_bar[d_idx]
        kappa_old = d_kappa[d_idx]
        kappa_o = d_kappa_o[d_idx]
        betad = d_betad[d_idx]
        alphad = d_alphad[d_idx]
        lambda_d = d_lambda_d[d_idx]

        # KKT conditions to find kappa
        if eps_eq_bar > kappa_o:
            if eps_eq_bar > kappa_old:
                kappa = eps_eq_bar
        else:
            kappa = kappa_o

        # Calculate the Damage variable
        dam_hist = d_dam[d_idx]
        deltakappa = kappa - kappa_o
        dam = 1.0 - (kappa_o/kappa)*((1.0-alphad)+alphad*exp(-betad*deltakappa))
        if dam < 0 and deltakappa < 0 and abs(deltakappa)<1e-6:
            # printf("\n")
            # printf("deltakappa=%.9f dam=%.9f\n", deltakappa,dam)
            # printf("\n")
            dam = 0
            kappa = kappa_o

        d_kappa[d_idx] = kappa
        d_dam[d_idx] = dam
        delta_dam = dam - dam_hist
        k_bar = 0.001
        # g_dam = (1-k_bar)*pow((1-dam),2)+k_bar
        g_prime_dam = -2.0 *(1-k_bar)*(1-dam)

        # Flag particle to use trial values
        d_flag[d_idx] = 0

        # Consolidation parameters
        lambda_mcc = d_lambda_mcc[d_idx]
        kappa_mcc = d_kappa_mcc[d_idx]
        void_ratio = d_void_ratio[d_idx]
        void_ref = d_void_ref[d_idx]
        v= (1.0+void_ratio)/(lambda_mcc-kappa_mcc)

        # Initialize stress and stress rate tensors
        for i in range(9):
            sigma[i] = d_sigma_tr[idx + i]
            sig_dot[i] = d_sigma_dot[idx + i]
            spin_dot[i] = d_spin_dot[idx + i]
            
        if self.c_model > 0 :
            p = d_p[d_idx]
            q = d_q[d_idx]
            pc = d_pc[d_idx]
            pcn = pc
            ms = d_ms[d_idx]
            y = yield_criterion_mcc(q, p, pc, ms)

            # If yielding
            if y > 1e-8:

                # Trial hydrostatic stress and von Mises stress
                ptr = p
                qtr = q

                # Elastic constants
                k = d_bulk[d_idx]
                g = d_shear[d_idx]
                

                # Counter of iterations for NR
                count = 0

                # Initialize consistency parameter
                dg = 0.0

                # Newton-Raphson loop
                while y > self.tol and count < self.max_iter:

                    # Pre-calculate the partial derivatives of the yield surface w.r.t. the consistency parameter, dg (DeltaL_k in ConsUpdate.m Enrique Approximate CPP)
                    a = (-2*k * pow(dg * (2*p - pc) + 1/v, 2) * (p - pc/2)) / \
                        (8*k * pow(p - pc/2, 2) * pow(dg, 3) + 8*(k * (1/v) + p/2 - pc/4) * (p - pc/2) *
                        pow(dg, 2) + 2*(1/v) * (k * (1/v) + 2*p - pc - pcn/2) * dg + pow(1/v, 2))

                    b1 = -q / (dg + pow(ms, 2) / (6*g))

                    c1 = ((-2*p + pc) * (1/v) * pcn) / \
                        (8*k * pow(p - pc/2, 2) * pow(dg, 3) + 8*(k * (1/v) + p/2 - pc/4) * (p - pc/2) *
                        pow(dg, 2) + 2*(1/v) * (k * (1/v) + 2*p - pc - pcn/2) * dg + pow(1/v, 2))

                    # Derivative of the yield function (Fp_k)
                    dy = (2*p - pc) * a + (2*q / pow(ms, 2)) * b1 - p * c1

                    # Update the consistency parameter
                    dg -= y/dy

                    # Auxiliary values to find pc
                    b2 = (1/v) * (1 + 2*k * dg) + 2*dg * ptr
                    if dam > 0:
                        b2 -= g_prime_dam/lambda_d * (1/v) * (1 + 2*k * dg) * delta_dam
                    c2 = pcn * (1/v) * (1 + 2*k * dg)
                    d1 = (b2 / dg)
                    d2 = sqrt(pow(b2 / dg, 2) - (4*c2) / dg)

                    # Roots of second order polynomial used to determine pc
                    pc_a = .5*(d1 + d2)
                    pc_b = .5*(d1 - d2)

                    # Update pc
                    if pc_a < 0:
                        pc = pc_a 
                    else:
                        pc = pc_b

                    # Update p and q
                    p = (ptr + k*dg * pc) / (1 + 2*k * dg)
                    q = qtr / (1 + 6*g * dg / pow(ms, 2))

                    # Calculate if stress on the yield surface
                    y = yield_criterion_mcc(q, p, pc, ms)

                    # Increment iteration counter
                    count += 1

                # Check if convergence obtained
                if y > self.tol:
                    printf("\n")
                    printf("====== NR DID NOT CONVERGE AFTER ")
                    printf("%d",count)
                    printf("ITERATIONS ======")
                    printf('Residual F')
                    printf("%.16f\n", y)
                    printf('\n')

                d_pc[d_idx] = pc

                # Update bulk and shear moduli
                #if p != 0:
                #    void_ratio = void_ref - lambda_mcc * log(-pc/1000) + kappa_mcc * log(pc/p)
                #    k = abs((1+void_ratio)*p/kappa_mcc)
                #    g = 3*k * (1 - 2*self.nu) / (2*(1 + self.nu))
                #    d_bulk[d_idx] = k
                #    d_void_ratio[d_idx] = void_ratio
                #    d_shear[d_idx] = g

                # Rate of plastic multiplier
                dg_dot = dg / dt

                # Volumetric strain rate
                eps_dot_v = d_eps_dot[idx] + d_eps_dot[idx + 4] + \
                            d_eps_dot[idx + 8]

                # Construct deformation tensors
                norm_s = sqrt(2.0 / 3.0) * qtr
                for i in range(9):

                    if norm_s > 0:
                        # Deviatoric plastic flow directions
                        n_i = d_sigma_dev[idx + i] / norm_s
                    else:
                        n_i=0
                    
                    # d_n[idx+i] = n_i

                    # Total strain rate decomposition
                    eps_dot_d = d_eps_dot[idx + i]
                    if i % 4 == 0:
                        eps_dot_d -= eps_dot_v / 3.0

                    # Plastic strain rates
                    eps_p_dot_d = sqrt(6) * dg_dot * q * n_i / (ms * ms)
                    eps_p_dot_v = dg_dot * (2*p - pc)
                    eps_p_dot_h = 0.0
                    if i % 4 == 0:
                        eps_p_dot_h = eps_p_dot_v / 3.0
                    d_eps_p_dot[idx + i] = eps_p_dot_d + eps_p_dot_h

                    # Stress rate
                    eps_e_dot_d = eps_dot_d - eps_p_dot_d
                    eps_e_dot_v = eps_dot_v - eps_p_dot_v
                    p_i = 0.0
                    if i % 4 == 0:
                        p_i = k * eps_e_dot_v
                    sdot = p_i + 2 * g * eps_e_dot_d

                    # Small-strain stress rate
                    sig_dot[i] = sdot

                    # Small-strain updated stress
                    sigma[i] += sdot * dt

                # Update accumulated plastic strain and uniaxial yield stress
                d_ep_acc[d_idx] += dg

                # Flag particle to update stress and strain
                d_flag[d_idx] = 1
        # Update bulk and shear moduli
        if p != 0:
            void_ratio = void_ref - lambda_mcc * log(-pc/1000) + kappa_mcc * log(pc/p)
            k = abs((1+void_ratio)*p/kappa_mcc)
            # *(1-dam)
            g = 3*k * (1 - 2*self.nu) / (2*(1 + self.nu))
            d_bulk[d_idx] = k
            d_void_ratio[d_idx] = void_ratio
            d_shear[d_idx] = g

        # Jaumann stress rate (~ large deformation)
        matrix_multiply(sigma, spin_dot, sig_spin, 3)
        matrix_multiply(spin_dot, sigma, sig_spin_t, 3)

        for i in range(9):
            d_sigma_dot[idx + i] = sig_dot[i] - sig_spin[i] + \
                                   sig_spin_t[i]

        # Update stress invariants
        # TODO: Find out how to do this in the output rather than every time
        #   step here!!!
        p_dot = (d_sigma_dot[idx] + d_sigma_dot[idx + 4] +
                 d_sigma_dot[idx + 8]) / 3.0
        p_n = (d_sigma[idx] + d_sigma[idx + 4] + d_sigma[idx + 8]) / 3.0
        ph = p_n + p_dot * dt
        d_p[d_idx] = ph

        norm_s2 = 0.0
        for i in range(9):
            s_i = d_sigma[idx + i] + d_sigma_dot[idx + i] * dt
            if i % 4 == 0:
                s_i -= ph
            norm_s2 += s_i * s_i
        d_q[d_idx] = sqrt(3 * norm_s2 / 2)

        summ=0.0
        for i in range(9):
            summ += (d_eps_p_dot[i+idx]*dt)*(d_eps_p_dot[i+idx]*dt)
        d_eps_p_norm[d_idx] += summ
        
    def _get_helpers_(self):
        return[yield_criterion_mcc, matrix_multiply]

class ChemicalCoupledDamageModel(Equation):
    def __init__(self, dest, sources, nu=0.3, c_model=1, y_criterion=5, debug=0):
        self.nu = nu
        self.c_model = c_model
        self.y_criterion = y_criterion
        self.debug = debug
        super(ChemicalCoupledDamageModel, self).__init__(dest, sources)

    def initialize(self, d_idx, d_sigma, d_sigma_tr,
                   d_sigma_dot, d_eps_dot, d_eps, d_kd,d_kappa_o, 
                   d_betad, d_alphad, d_dam, d_dam_m,d_dam_c, d_rho,d_ad,d_m_inf,d_eps_o,d_etad, d_flag, d_gid, dt,t,d_eps_eq):

        i, idx, count = declare("int", 3)
        sig_dot = declare("matrix(9)", 1)

        # Index used to access values in arrays
        idx = int(9 * d_idx)

        # Flag particle to use trial values
        d_flag[d_idx] = 0

        # Initialize stress and stress rate tensors
        for i in range(9):
            sig_dot[i] = d_sigma_dot[idx + i]

        # Find strain invariants and equivalent strain
        I1= d_eps[idx] + d_eps[idx + 4] + d_eps[idx + 8]

        kd = d_kd[d_idx]
        kappa_o = d_kappa_o[d_idx]
        betad = d_betad[d_idx]
        alphad = d_alphad[d_idx]
        rho_s=d_rho[d_idx]
        ad=d_ad[d_idx]
        m_inf=d_m_inf[d_idx]
        eps_o=d_eps_o[d_idx]
        etad=d_etad[d_idx]
        #m_inf=-.02*t+.03
        #m_inf=(.01*t*t)+.01

        #previous timestep damages
        prev_dam_m=d_dam_m[d_idx]
        prev_dam_c=d_dam_c[d_idx]

        e2 = 0.0
        for i in range(9):
            e = d_eps[idx + i]
            if i % 4 == 0:
                e -= I1/3
            e2 += e*e
        q_eps = sqrt(3.0*e2 / 2.0)
        J2=q_eps*q_eps/3

        eps_eq=I1*(kd-1)/((2*kd)*(1-2*self.nu))+(1/(2*kd))*sqrt((pow(kd-1,2)/pow(1-2*self.nu,2))*pow(I1,2)+(J2*2*kd)/pow(1+self.nu,2))
        d_eps_eq[d_idx]=eps_eq
            #NOTE output the eps_eq
        # KKT conditions to find kappa
        if eps_eq > kappa_o:
            kappa=eps_eq
        else:
            kappa=kappa_o

        # chemical Damage threshold condition
        eps_vol=I1/3
        if eps_vol< eps_o:
            eps_vol=eps_o

        # Calculate the mechanical Damage variable
        deltakappa=kappa-kappa_o
        dam_m=1-(kappa_o/kappa)*((1-alphad)+alphad*exp(-betad*deltakappa))
        if dam_m>prev_dam_m:
            d_dam_m[d_idx]=dam_m
        else: # NOTE damage cannot be less than that of precious timestep
            d_dam_m[d_idx]=prev_dam_m

        #Calculate the chemical Damage variable
        deltaev=eps_vol-eps_o
        lambdad=1-exp(-ad*(deltaev))
        dam_c=etad*lambdad/rho_s*m_inf
        if dam_c>prev_dam_c:
            d_dam_c[d_idx]=dam_c
        else:
            d_dam_c[d_idx]=prev_dam_c
        
        #Calculate the total Damage variable
        dam=dam_m+dam_c
        d_dam[d_idx]=dam

        # Calculate the final stress rate post damage. Final stress integrated in integrator_steppers.py
        for i in range(9):
            d_sigma_dot[idx + i]=dam*sig_dot[i]

"""
This class is implementing the approximate CPP. 
It is using continuum mechanics sign convention!
"""

class ModifiedCamClay_Approx_CPP(Equation):
    # def __init__(self, nu=0.3, c_model=1, y_criterion=3, tol=1e-5, max_iter=100, debug=0, *args, **kwargs):
    def __init__(self, dest, sources, nu=0.3, c_model=1, tol=1e-5, max_iter=100, debug=0):
        self.nu = nu
        self.c_model = c_model
        # self.y_criterion = y_criterion
        self.tol = tol
        self.max_iter = max_iter
        self.debug = debug
        # super(ModifiedCamClay_Approx_CPP, self).__init__(*args, **kwargs)
        super(ModifiedCamClay_Approx_CPP, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_q, d_sigma, d_sigma_tr, d_sigma_dev,d_void_ratio,d_void_ref,
                   d_sigma_dot, d_eps_dot, d_spin_dot, d_eps_p_dot, d_ep_acc, d_pc,
                   d_lambda_mcc, d_kappa_mcc, d_ms, d_bulk, d_shear, d_flag, d_gid, dt):

        i, idx, count = declare("int", 3)
        sigma, sig_dot, sig_spin, sig_spin_t = declare("matrix(9)", 4)
        spin_dot = declare("matrix(9)")
        n = declare("matrix(9)")
        
        # Index used to access values in arrays
        idx = int(9 * d_idx)

        # Flag particle to use trial values
        d_flag[d_idx] = 0

        # Consolidation parameters
        lambda_mcc = d_lambda_mcc[d_idx]
        kappa_mcc = d_kappa_mcc[d_idx]
        void_ratio = d_void_ratio[d_idx]
        void_ref = d_void_ref[d_idx]
        v= (1+void_ratio)/(lambda_mcc-kappa_mcc)

        # Initialize stress and stress rate tensors
        for i in range(9):
            sigma[i] = d_sigma_tr[idx + i]
            sig_dot[i] = d_sigma_dot[idx + i]

        # If not elastic material
        if self.c_model > 0:
            p = d_p[d_idx]
            q = d_q[d_idx]
            pc = d_pc[d_idx]
            pcn = pc
            ms = d_ms[d_idx]

            # Initialize temporary vectors
            for i in range(9):
                sigma[i] = d_sigma[idx + i]
                spin_dot[i] = d_spin_dot[idx + i]

            # Check for yielding
            y = yield_criterion_mcc(q, p, pc, ms)

            # If yielding
            if y > 1e-8:

                # Trial hydrostatic stress and von Mises stress
                ptr = p
                qtr = q

                # Elastic constants
                k = d_bulk[d_idx]
                g = d_shear[d_idx]
                

                # Counter of iterations for NR
                count = 0

                # Initialize consistency parameter
                dg = 0.0

                # Newton-Raphson loop
                while y > self.tol and count < self.max_iter:

                    # Pre-calculate the partial derivatives of the yield surface w.r.t. the consistency parameter, dg (DeltaL_k in ConsUpdate.m Enrique Approximate CPP)
                    a = (-2*k * pow(dg * (2*p - pc) + 1/v, 2) * (p - pc/2)) / \
                        (8*k * pow(p - pc/2, 2) * pow(dg, 3) + 8*(k * (1/v) + p/2 - pc/4) * (p - pc/2) *
                         pow(dg, 2) + 2*(1/v) * (k * (1/v) + 2*p - pc - pcn/2) * dg + pow(1/v, 2))

                    b1 = -q / (dg + pow(ms, 2) / (6*g))

                    c1 = ((-2*p + pc) * (1/v) * pcn) / \
                        (8*k * pow(p - pc/2, 2) * pow(dg, 3) + 8*(k * (1/v) + p/2 - pc/4) * (p - pc/2) *
                         pow(dg, 2) + 2*(1/v) * (k * (1/v) + 2*p - pc - pcn/2) * dg + pow(1/v, 2))

                    # Derivative of the yield function (Fp_k)
                    dy = (2*p - pc) * a + (2*q / pow(ms, 2)) * b1 - p * c1

                    # Update the consistency parameter
                    dg -= y/dy

                    # Auxiliary values to find pc
                    b2 = (1/v) * (1 + 2*k * dg) + 2*dg * ptr
                    c2 = pcn * (1/v) * (1 + 2*k * dg)
                    d1 = (b2 / dg)
                    d2 = sqrt(pow(b2 / dg, 2) - (4*c2) / dg)

                    # Roots of second order polynomial used to determine pc
                    pc_a = .5*(d1 + d2)
                    pc_b = .5*(d1 - d2)

                    # Update pc
                    if pc_a < 0:
                        pc = pc_a
                    else:
                        pc = pc_b

                    # Update p and q
                    p = (ptr + k*dg * pc) / (1 + 2*k * dg)
                    q = qtr / (1 + 6*g * dg / pow(ms, 2))

                    # Calculate if stress on the yield surface
                    y = yield_criterion_mcc(q, p, pc, ms)

                    # Increment iteration counter
                    count += 1

                # Check if convergence obtained
                if y > self.tol:
                    printf("\n")
                    printf("====== NR DID NOT CONVERGE AFTER ")
                    printf("%d",count)
                    printf("ITERATIONS ======")
                    printf('Residual F')
                    printf("%.16f\n", y)
                    printf('\n')

                d_pc[d_idx] = pc

                # # Update bulk and shear moduli
                # if p != 0:
                #     void_ratio = void_ref - lambda_mcc * log(-pc/1000) + kappa_mcc * log(pc/p)
                #     k = abs((1+void_ratio)*p/kappa_mcc)
                #     g = 3*k * (1 - 2*self.nu) / (2*(1 + self.nu))
                #     d_bulk[d_idx] = k
                #     d_void_ratio[d_idx] = void_ratio
                #     d_shear[d_idx] = g

                # Rate of plastic multiplier
                dg_dot = dg / dt

                # Volumetric strain rate
                eps_dot_v = d_eps_dot[idx] + d_eps_dot[idx + 4] + \
                            d_eps_dot[idx + 8]

                # Construct deformation tensors
                norm_s = sqrt(2.0 / 3.0) * qtr
                for i in range(9):

                    if norm_s > 0:
                        # Deviatoric plastic flow directions
                        n_i = d_sigma_dev[idx + i] / norm_s
                    else:
                        n_i=0
                    
                    # d_n[idx+i] = n_i

                    # Total strain rate decomposition
                    eps_dot_d = d_eps_dot[idx + i]
                    if i % 4 == 0:
                        eps_dot_d -= eps_dot_v / 3.0

                    # Plastic strain rates
                    eps_p_dot_d = sqrt(6) * dg_dot * q * n_i / (ms * ms)
                    eps_p_dot_v = dg_dot * (2*p - pc)
                    eps_p_dot_h = 0.0
                    if i % 4 == 0:
                        eps_p_dot_h = eps_p_dot_v / 3.0
                    d_eps_p_dot[idx + i] = eps_p_dot_d + eps_p_dot_h

                    # Stress rate
                    eps_e_dot_d = eps_dot_d - eps_p_dot_d
                    eps_e_dot_v = eps_dot_v - eps_p_dot_v
                    p_i = 0.0
                    if i % 4 == 0:
                        p_i = k * eps_e_dot_v
                    sdot = p_i + 2 * g * eps_e_dot_d

                    # Small-strain stress rate
                    sig_dot[i] = sdot

                    # Small-strain updated stress
                    sigma[i] += sdot * dt

                # Update accumulated plastic strain and uniaxial yield stress
                d_ep_acc[d_idx] += dg

                # Flag particle to update stress and strain
                d_flag[d_idx] = 1

                if self.debug and d_gid[d_idx] == self.debug:
                    printf("\n")
                    printf("====== Called DPSolver ======")
                    printf('p trial')
                    printf("%.16f\n", d_p[d_idx])
                    printf('\n')
                    printf('q trial')
                    printf("%.16f\n", d_q[d_idx])
                    printf('\n')
        # Update bulk and shear moduli
        if p != 0:
            void_ratio = void_ref - lambda_mcc * log(-pc/1000) + kappa_mcc * log(pc/p)
            k = abs((1+void_ratio)*p/kappa_mcc)
            g = 3*k * (1 - 2*self.nu) / (2*(1 + self.nu))
            d_bulk[d_idx] = k
            d_void_ratio[d_idx] = void_ratio
            d_shear[d_idx] = g

        # Jaumann stress rate (~ large deformation)
        matrix_multiply(sigma, spin_dot, sig_spin, 3)
        matrix_multiply(spin_dot, sigma, sig_spin_t, 3)

        for i in range(9):
            d_sigma_dot[idx + i] = sig_dot[i] - sig_spin[i] + \
                                   sig_spin_t[i]

        # Update stress invariants
        # TODO: Find out how to do this in the output rather than every time
        #   step here!!!
        p_dot = (d_sigma_dot[idx] + d_sigma_dot[idx + 4] +
                 d_sigma_dot[idx + 8]) / 3.0
        p_n = (d_sigma[idx] + d_sigma[idx + 4] + d_sigma[idx + 8]) / 3.0
        ph = p_n + p_dot * dt
        d_p[d_idx] = ph

        norm_s2 = 0.0
        for i in range(9):
            s_i = d_sigma[idx + i] + d_sigma_dot[idx + i] * dt
            if i % 4 == 0:
                s_i -= ph
            norm_s2 += s_i * s_i
        d_q[d_idx] = sqrt(3 * norm_s2 / 2)

        if self.debug and d_gid[d_idx] == self.debug:
            printf("Stress rate")
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx], d_sigma_dot[idx + 1],
                   d_sigma_dot[idx + 2])
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx + 3],
                   d_sigma_dot[idx + 4], d_sigma_dot[idx + 5])
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx + 6],
                   d_sigma_dot[idx + 7], d_sigma_dot[idx + 8])
            printf("\n")

            printf("Plastic strain rate")
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx], d_eps_p_dot[idx + 1],
                   d_eps_p_dot[idx + 2])
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx + 3],
                   d_eps_p_dot[idx + 4], d_eps_p_dot[idx + 5])
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx + 6],
                   d_eps_p_dot[idx + 7], d_eps_p_dot[idx + 8])
            printf( "====================================")

    def _get_helpers_(self):
        return[yield_criterion_mcc, matrix_multiply]
    

    
"""
This class is implementing the exact CPP.
"""

class ModifiedCamClay_Exact_CPP(Equation):
    def __init__(self, dest, sources, nu=0.3, c_model=1, y_criterion=3, tol=1e-3, max_iter=500, debug=0):
        self.nu = nu
        self.c_model = c_model
        self.y_criterion = y_criterion
        self.tol = tol
        self.max_iter = max_iter
        self.debug = debug
        super(ModifiedCamClay_Exact_CPP, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_q, d_sigma, d_sigma_tr, d_sigma_dev,d_n_i,
                   d_sigma_dot, d_eps_dot, d_spin_dot, d_eps_p_dot, d_ep_acc, d_pc, d_void_ratio, d_void_ref,
                   d_kappa_mcc, d_lambda_mcc, d_ms, d_bulk, d_shear, d_flag, d_gid, dt):

        i, idx, count = declare("int", 3)
        sigma, sig_dot, sig_spin, sig_spin_t = declare("matrix(9)", 4)
        spin_dot = declare("matrix(9)")

        # Index used to access values in arrays
        idx = int(9 * d_idx)

        # Flag particle to use trial values
        d_flag[d_idx] = 0
        
        # Consolidation parameters
        lambda_mcc = d_lambda_mcc[d_idx]
        kappa_mcc = d_kappa_mcc[d_idx]
        void_ratio = d_void_ratio[d_idx]
        void_ref = d_void_ref[d_idx]
        v= (1+void_ratio)/(lambda_mcc-kappa_mcc)
        
        # Initialize stress and stress rate tensors
        for i in range(9):
            sigma[i] = d_sigma_tr[idx + i]
            sig_dot[i] = d_sigma_dot[idx + i]
            spin_dot[i] = d_spin_dot[idx + i]

        # If not elastic material
        if self.c_model > 0:
            # initialize p and q using the Trial hydrostatic stress and Trial von Mises stress
            p = d_p[d_idx]
            q = d_q[d_idx]
            pc = d_pc[d_idx]
            pcn = pc
            ms = d_ms[d_idx]

            # Check for yielding
            y = yield_criterion_mcc(q, p, pc, ms)

            # If yielding
            if y > 1e-8:

                # Trial hydrostatic stress and Trial von Mises stress
                p=-p
                pc=-pc
                pcn = pc
                ptr = p
                qtr = q

                # Elastic constants
                k = d_bulk[d_idx]
                g = d_shear[d_idx]

                # Counter of iterations for NR
                count = 0

                # Initialize consistency parameter
                dg = 0.0

                # Newton-Raphson loop
                while abs(y) > self.tol and count < self.max_iter:
                    # 
                    pc_j = pc

                    G_j = pcn * exp(v*dg*(2*ptr-pc_j)/(1+2*dg*k))-pc_j

                    while abs(G_j) > self.tol and count < self.max_iter:
                        # 
                        Gp_j = pcn * exp(v*dg*(2*ptr-pc_j)/(1+2*dg*k))*(-v*dg/(1+2*dg*k))-1

                        pc_j = pc_j - G_j/Gp_j

                        G_j = pcn * exp(v*dg*(2*ptr-pc_j)/(1+2*dg*k))-pc_j
                    
                    pc = pc_j

                    y = yield_criterion_mcc(q, p, pc, ms)

                    # Pre-calculate the partial derivatives of the yield surface w.r.t. the consistency parameter, dg (DeltaL_k in ConsUpdate.m Enrique Approximate CPP)
                    a = -k*(2*p-pc)/(1+(2*k+v*pc)*dg)

                    b = -q / (dg + pow(ms, 2) / (6*g))

                    c = v*pc*(2*p-pc)/(1+(2*k+v*pc)*dg)

                    # Derivative of the yield function (Fp_k)
                    dy = (2*p - pc) * a + (2*q / pow(ms, 2)) * b - p * c

                    # Update the consistency parameter
                    dg -= y/dy

                    # Update p and q
                    p = (ptr + k*dg * pc) / (1 + 2*k * dg)
                    q = qtr / (1 + 6*g * dg / pow(ms, 2))

                    # Calculate if stress on the yield surface
                    y = yield_criterion_mcc(q, p, pc, ms)

                    # Increment iteration counter
                    count += 1

                # Check if convergence obtained
                if y > self.tol:
                    printf("\n")
                    printf("====== NR DID NOT CONVERGE AFTER ")
                    printf("%d",count)
                    printf("ITERATIONS ======")
                    printf('Residual F')
                    printf("%.16f\n", y)
                    printf('\n')

                # change the sign of p,pc back
                p = -p
                pc = -pc
                d_pc[d_idx] = pc

                # Update bulk and shear moduli
                if abs(p) > 0.01:
                    void_ratio = void_ref - lambda_mcc * log(-pc/1000) + kappa_mcc * log(pc/p)
                    k = abs((1+void_ratio)*p/kappa_mcc)
                    g = 3*k * (1 - 2*self.nu) / (2*(1 + self.nu))
                    d_bulk[d_idx] = k
                    d_void_ratio[d_idx] = void_ratio
                    d_shear[d_idx] = g

                # Rate of plastic multiplier
                dg_dot = dg / dt

                # Volumetric strain rate
                eps_dot_v = d_eps_dot[idx] + d_eps_dot[idx + 4] + \
                            d_eps_dot[idx + 8]

                # Construct deformation tensors
                norm_s = sqrt(2.0 / 3.0) * qtr
                for i in range(9):
                    # n_i = d_sigma_dev[idx + i] / norm_s
                    if norm_s > 0:
                        # Deviatoric plastic flow directions
                        n_i = d_sigma_dev[idx + i] / norm_s
                    else:
                        n_i=0

                    # Total strain rate decomposition
                    eps_dot_d = d_eps_dot[idx + i]
                    if i % 4 == 0:
                        eps_dot_d -= eps_dot_v / 3.0

                    # Plastic strain rates
                    eps_p_dot_d = sqrt(6) * dg_dot * q * n_i / (ms * ms)
                    eps_p_dot_v = dg_dot * (2*p - pc)
                    eps_p_dot_h = 0.0
                    if i % 4 == 0:
                        eps_p_dot_h = eps_p_dot_v / 3.0
                    d_eps_p_dot[idx + i] = eps_p_dot_d + eps_p_dot_h

                    # Stress rate
                    eps_e_dot_d = eps_dot_d - eps_p_dot_d
                    eps_e_dot_v = eps_dot_v - eps_p_dot_v
                    p_i = 0.0
                    if i % 4 == 0:
                        p_i = k * eps_e_dot_v
                    sdot = p_i + 2 * g * eps_e_dot_d

                    # Small-strain stress rate
                    sig_dot[i] = sdot

                    # Small-strain updated stress
                    sigma[i] += sdot * dt

                # Update accumulated plastic strain and uniaxial yield stress
                d_ep_acc[d_idx] += dg

                # Flag particle to update stress and strain
                d_flag[d_idx] = 1

                if self.debug and d_gid[d_idx] == self.debug:
                    printf("\n")
                    printf("====== Called DPSolver ======")
                    printf('p trial')
                    printf("%.16f\n", d_p[d_idx])
                    printf('\n')
                    printf('q trial')
                    printf("%.16f\n", d_q[d_idx])
                    printf('\n')
        # Update bulk and shear moduli
        if abs(p) > 0.01:
            void_ratio = void_ref - lambda_mcc * log(-pc/1000) + kappa_mcc * log(pc/p)
            k = abs((1+void_ratio)*p/kappa_mcc)
            g = 3*k * (1 - 2*self.nu) / (2*(1 + self.nu))
            d_bulk[d_idx] = k
            d_void_ratio[d_idx] = void_ratio
            d_shear[d_idx] = g

        # Jaumann stress rate (~ large deformation)
        matrix_multiply(sigma, spin_dot, sig_spin, 3)
        matrix_multiply(spin_dot, sigma, sig_spin_t, 3)

        for i in range(9):
            d_sigma_dot[idx + i] = sig_dot[i] - sig_spin[i] + \
                                   sig_spin_t[i]

        # Update stress invariants
        # TODO: Find out how to do this in the output rather than every time
        #   step here!!!
        p_dot = (d_sigma_dot[idx] + d_sigma_dot[idx + 4] +
                 d_sigma_dot[idx + 8]) / 3.0
        p_n = (d_sigma[idx] + d_sigma[idx + 4] + d_sigma[idx + 8]) / 3.0
        ph = p_n + p_dot * dt
        d_p[d_idx] = ph

        norm_s2 = 0.0
        for i in range(9):
            s_i = d_sigma[idx + i] + d_sigma_dot[idx + i] * dt
            if i % 4 == 0:
                s_i -= ph
            norm_s2 += s_i * s_i
        d_q[d_idx] = sqrt(3 * norm_s2 / 2)

        if self.debug and d_gid[d_idx] == self.debug:
            printf("Stress rate")
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx], d_sigma_dot[idx + 1],
                   d_sigma_dot[idx + 2])
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx + 3],
                   d_sigma_dot[idx + 4], d_sigma_dot[idx + 5])
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx + 6],
                   d_sigma_dot[idx + 7], d_sigma_dot[idx + 8])
            printf("\n")

            printf("Plastic strain rate")
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx], d_eps_p_dot[idx + 1],
                   d_eps_p_dot[idx + 2])
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx + 3],
                   d_eps_p_dot[idx + 4], d_eps_p_dot[idx + 5])
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx + 6],
                   d_eps_p_dot[idx + 7], d_eps_p_dot[idx + 8])
            printf( "====================================")

    def _get_helpers_(self):
        return[yield_criterion_mcc, matrix_multiply]

"""
This class is implementing the (exact) CRM. 
First, read data in the continuum sign convention, transfer to soil convention.
Finally, output the data back to the continuum sign convention.
"""

class ModifiedCamClayCRM(Equation):
    def __init__(self, dest, sources, nu=0.3, c_model=1, y_criterion=3, tol=1e-5, max_iter=500, debug=0):
        self.nu = nu
        self.c_model = c_model
        self.y_criterion = y_criterion
        self.tol = tol
        self.max_iter = max_iter
        self.debug = debug
        super(ModifiedCamClayCRM, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_q, d_sigma, d_sigma_tr, d_sigma_dev, d_ep_acc_d, d_ep_acc_v,
                   d_sigma_dot, d_eps_dot, d_spin_dot, d_eps_p_dot, d_ep_acc, d_pc, d_void_ratio, d_void_ref,
                   d_kappa_mcc, d_lambda_mcc, d_ms, d_bulk, d_shear, d_flag, d_gid, dt):

        i, idx, count = declare("int", 3)
        sigma, sig_dot, sig_spin, sig_spin_t = declare("matrix(9)", 4)
        spin_dot = declare("matrix(9)")

        # Index used to access values in arrays
        idx = int(9 * d_idx)

        # Flag particle to use trial values
        d_flag[d_idx] = 0
        
        # Consolidation parameters
        lambda_mcc = d_lambda_mcc[d_idx]
        kappa_mcc = d_kappa_mcc[d_idx]
        void_ratio = d_void_ratio[d_idx]
        void_ref = d_void_ref[d_idx]
        v= (1+void_ratio)/(lambda_mcc-kappa_mcc)
        
        
        # Initialize stress and stress rate tensors
        for i in range(9):
            sigma[i] = d_sigma_tr[idx + i]
            sig_dot[i] = d_sigma_dot[idx + i]

        # If not elastic material
        if self.c_model > 0:
            p = -d_p[d_idx]
            q = d_q[d_idx]
            pc = -d_pc[d_idx]
            pcn = pc
            ms = d_ms[d_idx]

            # Initialize temporary vectors
            for i in range(9):
                sigma[i] = d_sigma[idx + i]
                spin_dot[i] = d_spin_dot[idx + i]

            # Check for yielding
            y = yield_criterion_mcc(q, p, pc, ms)

            # If yielding
            if y > 1e-8:

                # Trial hydrostatic stress and von Mises stress
                ptr = p
                qtr = q

                # Elastic constants
                k = d_bulk[d_idx]
                g = d_shear[d_idx]


                # Counter of iterations for NR
                count = 0

                # Initialize consistency parameter
                dEpv = 0.0

                # Newton-Raphson loop
                while abs(y) > self.tol and count < self.max_iter:

                    # Pre-calculate the partial derivatives of the yield surface w.r.t. the consistency parameter, dg
                    a = -k
                    b = (-qtr/(ptr-.5*pc))*(k+.5*pc*v*(ptr-p)/(ptr-.5*pc))
                    # if abs(ptr-0.5*pc) < 1e-10:
                    #     # Handle division by zero
                    #     # ...
                    #     b = 0
                    # else:
                    #     # Perform division
                    #     # ...
                    #     b = (-qtr/(ptr-.5*pc))*(k+.5*pc*v*(ptr-p)/(ptr-.5*pc))
                    
                    c = v*pc

                    # Derivative of the yield function
                    dy = (2*p - pc) * a + (2*q / pow(ms, 2)) * b - p * c

                    # Update the consistency parameter
                    dEpv -= y/dy


                    # Update p, pc, and q
                    p = ptr -k*dEpv
                    pc=pcn*exp(v*dEpv)
                    if abs(ptr-0.5*pc) < 1e-10:
                        # Handle division by zero
                        # ...
                        q = qtr
                    else:
                        # Perform division
                        # ...
                        q = qtr*(p-.5*pc)/(ptr-.5*pc)

                    # Calculate if stress on the yield surface
                    y = yield_criterion_mcc(q, p, pc, ms)

                    # Increment iteration counter
                    count += 1

                # Check if convergence obtained
                if abs(y) > self.tol:
                    printf("\n")
                    printf("====== NR DID NOT CONVERGE AFTER ")
                    printf(" %d", count)
                    printf(" ITERATIONS ======")
                    printf("\n")
                    printf('Residual F')
                    printf("\n")
                    printf("%.16f\n", y)
                    printf('\n')
                    
            # change the sign of p,pc
            p = -p
            pc = -pc
            d_pc[d_idx] = pc

            # Update bulk and shear moduli
            if p != 0:
                void_ratio = void_ref - lambda_mcc * log(-pc/1000) + kappa_mcc * log(pc/p)
                k = abs((1+void_ratio)*p/kappa_mcc)
                g = 3*k * (1 - 2*self.nu) / (2*(1 + self.nu))
                d_bulk[d_idx] = k
                d_void_ratio[d_idx] = void_ratio
                d_shear[d_idx] = g

            # Volumetric strain rate
            eps_dot_v = d_eps_dot[idx] + d_eps_dot[idx + 4] + \
                        d_eps_dot[idx + 8]

            # Construct deformation tensors
            norm_s = sqrt(2.0 / 3.0) * qtr
            for i in range(9):

                # Deviatoric plastic flow directions
                if norm_s > 0:
                    n_i = d_sigma_dev[idx + i] / norm_s
                else:
                    n_i= 0

                # Total strain rate decomposition
                eps_dot_d = d_eps_dot[idx + i]
                if i % 4 == 0:
                    eps_dot_d -= eps_dot_v / 3.0

                # # Plastic strain rates
                # eps_p_dot_v = (dEpv/dt)
                # eps_p_dot_d = sqrt(6)*(dEpv/dt)*q*(k/3/g)/(2*p-pc)*n_i
                
                # eps_p_dot_h = 0.0
                # if i % 4 == 0:
                #     eps_p_dot_h = eps_p_dot_v / 3.0
                # d_eps_p_dot[idx + i] = eps_p_dot_d + eps_p_dot_h

                # # Stress rate
                # eps_e_dot_d = eps_dot_d - eps_p_dot_d
                # eps_e_dot_v = eps_dot_v - eps_p_dot_v
                # p_i = 0.0
                # if i % 4 == 0:
                #     p_i = k * eps_e_dot_v
                # sdot = p_i + 2 * g * eps_e_dot_d

                eps_e_dot_d = sqrt(2/3)*(q/2/g)*n_i/dt
                eps_p_dot_d = eps_dot_d - eps_e_dot_d
                p_i = 0.0
                if i % 4 == 0:
                    p_i = p
                sdot = (p+sqrt(2/3)*q*n_i)/dt


                # Small-strain stress rate
                sig_dot[i] = sdot

                # Small-strain updated stress
                sigma[i] += sdot * dt

            # Update accumulated plastic strain volumetric and deviatoric parts
            d_ep_acc_v[d_idx] += dEpv
            d_ep_acc_d[d_idx] += eps_p_dot_d*dt

            # Flag particle to update stress and strain
            d_flag[d_idx] = 1

            if self.debug and d_gid[d_idx] == self.debug:
                printf("\n")
                printf("====== Called DPSolver ======")
                printf('p trial')
                printf("%.16f\n", d_p[d_idx])
                printf('\n')
                printf('q trial')
                printf("%.16f\n", d_q[d_idx])
                printf('\n')

        # # Jaumann stress rate (~ large deformation)
        # matrix_multiply(sigma, spin_dot, sig_spin, 3)
        # matrix_multiply(spin_dot, sigma, sig_spin_t, 3)

        for i in range(9):
            d_sigma_dot[idx + i] = sig_dot[i] 
            # - sig_spin[i] + sig_spin_t[i]

        # Update stress invariants
        # TODO: Find out how to do this in the output rather than every time
        #   step here!!!
        # p_dot = (d_sigma_dot[idx] + d_sigma_dot[idx + 4] +
        #          d_sigma_dot[idx + 8]) / 3.0
        # p_n = (d_sigma[idx] + d_sigma[idx + 4] + d_sigma[idx + 8]) / 3.0
        # ph = p_n + p_dot * dt
        ph = p
        d_p[d_idx] = ph

        # norm_s2 = 0.0
        # for i in range(9):
        #     s_i = d_sigma[idx + i] + d_sigma_dot[idx + i] * dt
        #     if i % 4 == 0:
        #         s_i -= ph
        #     norm_s2 += s_i * s_i
        # d_q[d_idx] = sqrt(3 * norm_s2 / 2)
        d_q[d_idx] = q

        if self.debug and d_gid[d_idx] == self.debug:
            printf("Stress rate")
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx], d_sigma_dot[idx + 1],
                   d_sigma_dot[idx + 2])
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx + 3],
                   d_sigma_dot[idx + 4], d_sigma_dot[idx + 5])
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx + 6],
                   d_sigma_dot[idx + 7], d_sigma_dot[idx + 8])
            printf("\n")

            printf("Plastic strain rate")
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx], d_eps_p_dot[idx + 1],
                   d_eps_p_dot[idx + 2])
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx + 3],
                   d_eps_p_dot[idx + 4], d_eps_p_dot[idx + 5])
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx + 6],
                   d_eps_p_dot[idx + 7], d_eps_p_dot[idx + 8])
            printf( "====================================")

    def _get_helpers_(self):
        return[yield_criterion_mcc, matrix_multiply]

"""
This class is implementing the exact tensile CPP.
"""

class ModifiedCamClay_Exact_Tensile_CPP(Equation):
    def __init__(self, dest, sources, nu=0.3, c_model=1, y_criterion=3, tol=1e-5, max_iter=100, debug=0):
        self.nu = nu
        self.c_model = c_model
        self.y_criterion = y_criterion
        self.tol = tol
        self.max_iter = max_iter
        self.debug = debug
        super(ModifiedCamClay_Exact_Tensile_CPP, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_q, d_sigma, d_sigma_tr, d_sigma_dev,
                   d_sigma_dot, d_eps_dot, d_spin_dot, d_eps_p_dot, d_ep_acc, d_pc, d_void_ratio, d_void_ref,
                   d_kappa_mcc, d_lambda_mcc, d_ms, d_bulk, d_shear, d_flag, d_gid, dt,d_pt):

        i, idx, count = declare("int", 3)
        sigma, sig_dot, sig_spin, sig_spin_t = declare("matrix(9)", 4)
        spin_dot = declare("matrix(9)")

        # Index used to access values in arrays
        idx = int(9 * d_idx)

        # Flag particle to use trial values
        d_flag[d_idx] = 0
        
        # Consolidation parameters
        pt = d_pt[d_idx]
        lambda_mcc = d_lambda_mcc[d_idx]
        kappa_mcc = d_kappa_mcc[d_idx]
        void_ratio = d_void_ratio[d_idx]
        void_ref = d_void_ref[d_idx]
        v= (1+void_ratio)/(lambda_mcc-kappa_mcc)
        
        # Initialize stress and stress rate tensors
        for i in range(9):
            sigma[i] = d_sigma_tr[idx + i]
            sig_dot[i] = d_sigma_dot[idx + i]

        # If not elastic material
        if self.c_model > 0:
            # initialize p and q using the Trial hydrostatic stress and Trial von Mises stress
            p = d_p[d_idx]
            q = d_q[d_idx]
            pc = d_pc[d_idx]
            pcn = pc
            ms = d_ms[d_idx]

            # Initialize temporary vectors
            for i in range(9):
                # sigma[i] = d_sigma[idx + i]
                spin_dot[i] = d_spin_dot[idx + i]

            # Check for yielding
            y = yield_criterion_mcc_tensile(q, p, pc, pt, ms)

            # If yielding
            if y > 1e-8:

                # Trial hydrostatic stress and Trial von Mises stress
                p=-p
                pc=-pc
                pt=-pt
                pcn = pc
                ptr = p
                qtr = q

                # Elastic constants
                k = d_bulk[d_idx]
                g = d_shear[d_idx]

                # Counter of iterations for NR
                count = 0

                # Initialize consistency parameter
                dg = 0.0

                # Newton-Raphson loop
                while abs(y) > self.tol and count < self.max_iter:
                    # 
                    pc_j = pc

                    G_j = pcn * exp(v*dg*(2*ptr-pc_j-pt)/(1+2*dg*k))-pc_j

                    while abs(G_j) > self.tol and count < self.max_iter:
                        # 
                        Gp_j = pcn * exp(v*dg*(2*ptr-pc_j-pt)/(1+2*dg*k))*(-v*dg/(1+2*dg*k))-1

                        pc_j = pc_j - G_j/Gp_j

                        G_j = pcn * exp(v*dg*(2*ptr-pc_j-pt)/(1+2*dg*k))-pc_j
                    
                    pc = pc_j

                    y = yield_criterion_mcc_tensile(q, p, pc, pt, ms)

                    if abs(y) > self.tol and count < self.max_iter:
                        # Pre-calculate the partial derivatives of the yield surface w.r.t. the consistency parameter, dg (DeltaL_k in ConsUpdate.m Enrique Approximate CPP)
                        a = -k*(2*p-pc-pt)/(1+(2*k+v*pc)*dg)

                        b = -q / (dg + pow(ms, 2) / (6*g))

                        c = v*pc*(2*p-pc-pt)/(1+(2*k+v*pc)*dg)

                        # Derivative of the yield function (Fp_k)
                        dy = (2*p - pc-pt) * a + (2*q / pow(ms, 2)) * b + (pt-p) * c

                        # Update the consistency parameter
                        dg -= y/dy

                    # Update p and q
                    p = (ptr + k*dg * (pc+pt)) / (1 + 2*k * dg)
                    q = qtr / (1 + 6*g * dg / pow(ms, 2))

                    # Calculate if stress on the yield surface
                    y = yield_criterion_mcc_tensile(q, p, pc, pt, ms)

                    # Increment iteration counter
                    count += 1

                # Check if convergence obtained
                if y > self.tol:
                    printf("\n")
                    printf("====== NR DID NOT CONVERGE AFTER ")
                    printf("%d",count)
                    printf("ITERATIONS ======")
                    printf('Residual F')
                    printf("%.16f\n", y)
                    printf('\n')

                # change the sign of p,pc back
                p = -p
                pc = -pc
                pt = -pt
                d_pc[d_idx] = pc

                # if p is nonzero then Update bulk and shear moduli
                if abs(p) > self.tol:
                    void_ratio = void_ref - lambda_mcc * log(-pc/1000) + kappa_mcc * log(pc/p)
                    k = abs((1+void_ratio)*p/kappa_mcc)
                    g = 3*k * (1 - 2*self.nu) / (2*(1 + self.nu))
                    d_bulk[d_idx] = k
                    d_void_ratio[d_idx] = void_ratio
                    d_shear[d_idx] = g

                # Rate of plastic multiplier
                dg_dot = dg / dt

                # Volumetric strain rate
                eps_dot_v = d_eps_dot[idx] + d_eps_dot[idx + 4] + \
                            d_eps_dot[idx + 8]

                # Construct deformation tensors
                norm_s = sqrt(2.0 / 3.0) * qtr
                for i in range(9):
                
                    if norm_s > 0:
                        # Deviatoric plastic flow directions
                        n_i = d_sigma_dev[idx + i] / norm_s
                    else:
                        n_i=0

                    # Total strain rate decomposition
                    eps_dot_d = d_eps_dot[idx + i]
                    if i % 4 == 0:
                        eps_dot_d -= eps_dot_v / 3.0

                    # Plastic strain rates
                    eps_p_dot_d = sqrt(6) * dg_dot * q * n_i / (ms * ms)
                    eps_p_dot_v = dg_dot * (2*p - pc - pt)
                    eps_p_dot_h = 0.0
                    if i % 4 == 0:
                        eps_p_dot_h = eps_p_dot_v / 3.0
                    d_eps_p_dot[idx + i] = eps_p_dot_d + eps_p_dot_h

                    # Stress rate
                    eps_e_dot_d = eps_dot_d - eps_p_dot_d
                    eps_e_dot_v = eps_dot_v - eps_p_dot_v
                    p_i = 0.0
                    if i % 4 == 0:
                        p_i = k * eps_e_dot_v
                    sdot = p_i + 2 * g * eps_e_dot_d

                    # Small-strain stress rate
                    sig_dot[i] = sdot

                    # Small-strain updated stress
                    sigma[i] += sdot * dt

                # Update accumulated plastic strain and uniaxial yield stress
                d_ep_acc[d_idx] += dg

                # Flag particle to update stress and strain
                d_flag[d_idx] = 1

                if self.debug and d_gid[d_idx] == self.debug:
                    printf("\n")
                    printf("====== Called DPSolver ======")
                    printf('p trial')
                    printf("%.16f\n", d_p[d_idx])
                    printf('\n')
                    printf('q trial')
                    printf("%.16f\n", d_q[d_idx])
                    printf('\n')
        # Update bulk and shear moduli
        if abs(p) > self.tol:
            void_ratio = void_ref - lambda_mcc * log(-pc/1000) + kappa_mcc * log(pc/p)
            k = abs((1+void_ratio)*p/kappa_mcc)
            g = 3*k * (1 - 2*self.nu) / (2*(1 + self.nu))
            d_bulk[d_idx] = k
            d_void_ratio[d_idx] = void_ratio
            d_shear[d_idx] = g

        # Jaumann stress rate (~ large deformation)
        matrix_multiply(sigma, spin_dot, sig_spin, 3)
        matrix_multiply(spin_dot, sigma, sig_spin_t, 3)

        for i in range(9):
            d_sigma_dot[idx + i] = sig_dot[i] - sig_spin[i] + \
                                   sig_spin_t[i]

        # Update stress invariants
        # TODO: Find out how to do this in the output rather than every time
        #   step here!!!
        p_dot = (d_sigma_dot[idx] + d_sigma_dot[idx + 4] +
                 d_sigma_dot[idx + 8]) / 3.0
        p_n = (d_sigma[idx] + d_sigma[idx + 4] + d_sigma[idx + 8]) / 3.0
        ph = p_n + p_dot * dt
        d_p[d_idx] = ph

        norm_s2 = 0.0
        for i in range(9):
            s_i = d_sigma[idx + i] + d_sigma_dot[idx + i] * dt
            if i % 4 == 0:
                s_i -= ph
            norm_s2 += s_i * s_i
        d_q[d_idx] = sqrt(3 * norm_s2 / 2)

        if self.debug and d_gid[d_idx] == self.debug:
            printf("Stress rate")
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx], d_sigma_dot[idx + 1],
                   d_sigma_dot[idx + 2])
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx + 3],
                   d_sigma_dot[idx + 4], d_sigma_dot[idx + 5])
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx + 6],
                   d_sigma_dot[idx + 7], d_sigma_dot[idx + 8])
            printf("\n")

            printf("Plastic strain rate")
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx], d_eps_p_dot[idx + 1],
                   d_eps_p_dot[idx + 2])
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx + 3],
                   d_eps_p_dot[idx + 4], d_eps_p_dot[idx + 5])
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx + 6],
                   d_eps_p_dot[idx + 7], d_eps_p_dot[idx + 8])
            printf( "====================================")

    def _get_helpers_(self):
        return[yield_criterion_mcc_tensile, matrix_multiply]

"""
This class is implementing the Drucker Prager.
"""

class DruckerPragerSolverExact(Equation):

    def __init__(self, dest, sources, c_model=1, y_criterion=2, debug=0):
        self.c_model = c_model
        self.y_criterion = y_criterion
        self.debug = debug
        super(DruckerPragerSolverExact, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_q, d_sigma, d_sigma_tr, d_sigma_dev,
                   d_sigma_dot, d_eps_dot, d_spin_dot, d_eps_p_dot, d_ep_acc,
                   d_aphi, d_apsi, d_ac, d_sy, d_eta, d_h_mod, d_bulk, d_shear,
                   d_flag, d_gid, dt):

        i, idx = declare("int", 2)
        p, q = declare("matrix(1)", 2)
        sigma, sig_dot, sig_spin, sig_spin_t = declare("matrix(9)", 4)
        spin_dot = declare("matrix(9)")

        # Index used to access values in arrays
        idx = int(9*d_idx)

        # Flag particle to use trial values
        d_flag[d_idx] = 0

        # Initialize stress and stress rate tensors
        for i in range(9):
            sigma[i] = d_sigma_tr[idx + i]
            sig_dot[i] = d_sigma_dot[idx + i]

        # If not elastic material
        if self.c_model > 0:

            p[0] = d_p[d_idx]
            q[0] = d_q[d_idx]
            aphi = 0.0
            apsi = 0.0
            ac = d_ac[d_idx]
            sy = d_sy[d_idx]
            h_mod = 0.0

            # Initialize temporary vectors
            for i in range(9):
                sigma[i] = d_sigma[idx + i]
                spin_dot[i] = d_spin_dot[idx + i]

            # Drucker-Prager parameters
            if self.y_criterion == 2:
                aphi = d_aphi[d_idx]
                apsi = d_apsi[d_idx]

            # Check for yielding
            y = yield_criterion(q, p, aphi, ac, sy, self.y_criterion)

            # If yielding
            if y > 1e-6:

                # Elastic constants
                k = d_bulk[d_idx]
                g = d_shear[d_idx]

                # Plastic modulus
                if sy > 0:  # No hardening if no cohesion
                    h_mod = d_h_mod[d_idx]

                # Plastic multiplier 
                #######################################
                # (A.17 on P156, delta_lambda) (typo, it should be dgamma)
                dgamma = y / (2*g + k*aphi*apsi + ac*ac*h_mod) 

                # Volumetric strain rate
                eps_dot_v = d_eps_dot[idx] + d_eps_dot[idx + 4] + \
                            d_eps_dot[idx + 8]

                # Check if return to the cone is valid
                norm_s = sqrt(2.0/3.0)*q[0]
                if norm_s - 2*g*dgamma < 0.0:

                    # Initialize variables for apex calculations
                    eps_e_dot_v = 0.0
                    dgamma = 0.0
                    pn = (sigma[0] + sigma[4] + sigma[8]) / 3.0

                    if apsi != 0.0:

                        # Plastic multiplier and rate 
                        ##################################
                        # ALERT: Possible Error here
                        ##################################
                        # Eq (A.21)
                        dgamma = (aphi*p[0] - sy) / \
                                 (k*aphi*apsi + ac*ac*h_mod)
                        dg_dot = dgamma / dt

                        # Volumetric plastic strain rate
                        eps_p_dot_v = dg_dot*apsi

                        # Volumetric and deviatoric elastic strain rate
                        eps_e_dot_v = eps_dot_v - eps_p_dot_v

                        # Update stress and strain rates
                        for i in range(9):

                            # Get rid of spin rate
                            spin_dot[i] = 0.0

                            # Update plastic strain rate
                            ep_dot = d_eps_dot[idx + i]
                            if i % 4 == 0:
                                ep_dot = eps_p_dot_v / 3.0
                            d_eps_p_dot[idx + i] = ep_dot

                            # Updated small strain stress
                            dp = k*eps_e_dot_v*dt
                            s_i = -sigma[i]
                            if i % 4 == 0:
                                s_i += pn + dp
                            sigma[i] += s_i

                            # Updated stress rate
                            sig_dot[i] = s_i/dt

                    else:
                        # If dilation angle is zero, the particle is treated as
                        #  if all excess deformation is plastic and the maximum
                        #  stress is: p_max = ac*sy/aphi
                        for i in range(9):

                            p_i = 0.0
                            p_max = ac * sy / aphi  # Maximum tensile stress
                            eps_e_dot = 0.0
                            eps_e_dot_v = (p_max - pn) / (k * dt)
                            if i % 4 == 0:
                                p_i = p_max
                                eps_e_dot = eps_e_dot_v / 3.0

                            sig_dot[i] = (p_i - sigma[i]) / dt
                            spin_dot[i] = 0.0
                            d_eps_p_dot[idx + i] = d_eps_dot[idx + i] - \
                                                   eps_e_dot

                # Return to the smooth part of the cone
                else:

                    # Rate of plastic multiplier
                    dg_dot = dgamma / dt

                    for i in range(9):

                        # Deviatoric plastic flow directions
                        n_i = d_sigma_dev[idx + i] / norm_s

                        # Total strain rate decomposition
                        eps_dot_d = d_eps_dot[idx + i]
                        if i % 4 == 0:
                            eps_dot_d -= eps_dot_v/3.0

                        # Plastic strain rate
                        eps_p_dot_d = dg_dot*n_i
                        eps_p_dot_v = dg_dot*apsi
                        eps_p_dot_h = 0.0
                        if i % 4 == 0:
                            eps_p_dot_h = eps_p_dot_v / 3.0
                        d_eps_p_dot[idx + i] = eps_p_dot_d + eps_p_dot_h

                        # Stress rate
                        eps_e_dot_d = eps_dot_d - eps_p_dot_d
                        eps_e_dot_v = eps_dot_v - eps_p_dot_v
                        p_i = 0.0
                        if i % 4 == 0:
                            p_i = k*eps_e_dot_v
                        sdot = p_i + 2*g*eps_e_dot_d

                        # Small-strain stress rate
                        sig_dot[i] = sdot

                        # Small-strain updated stress
                        sigma[i] += sdot*dt

                # Update accumulated plastic strain and uniaxial yield stress
                ##############################################
                # Eq 2.58
                dlamb = dgamma*ac 
                d_ep_acc[d_idx] += dlamb
                d_sy[d_idx] += h_mod*dlamb

                # Flag particle to update stress and strain
                d_flag[d_idx] = 1

                if self.debug and d_gid[d_idx] == self.debug:
                    printf("\n")
                    printf("====== Called DPSolver ======")
                    printf('p trial')
                    printf("%.16f\n", d_p[d_idx])
                    printf('\n')
                    printf('q trial')
                    printf("%.16f\n", d_q[d_idx])
                    printf('\n')

                    printf('Sy_n+1,tr')
                    printf("%.16f\n", sy)
                    printf('\n')
                    printf('Sy_n+1')
                    printf("%.16f\n", d_sy[d_idx])
                    printf('\n')
                    printf('aphi, apsi, ac, ac*sy, ac*sy/aphi')
                    printf("%.6f %.6f %.6f %.6f %.9f\n", aphi, apsi, ac, ac*sy,
                           ac*sy/aphi)
                    printf('\n')

                    printf('Y_tr')
                    printf("%.6f\n", y)
                    printf('\n')

        # Jaumann stress rate (~ large deformation)
        matrix_multiply(sigma, spin_dot, sig_spin, 3)
        matrix_multiply(spin_dot, sigma, sig_spin_t, 3)

        for i in range(9):
            d_sigma_dot[idx + i] = sig_dot[i] - sig_spin[i] + \
                                   sig_spin_t[i]

        # Update stress invariants
        # TODO: Find out how to do this in the output rather than every time
        #   step here!!!
        p_dot = (d_sigma_dot[idx] + d_sigma_dot[idx + 4] +
                 d_sigma_dot[idx + 8]) / 3.0
        p_n = (d_sigma[idx] + d_sigma[idx + 4] + d_sigma[idx + 8]) / 3.0
        ph = p_n + p_dot*dt
        d_p[d_idx] = ph

        norm_s2 = 0.0
        for i in range(9):
            s_i = d_sigma[idx + i] + d_sigma_dot[idx + i]*dt
            if i % 4 == 0:
                s_i -= ph
            norm_s2 += s_i * s_i
        d_q[d_idx] = sqrt(3*norm_s2 / 2)

        if self.debug and d_gid[d_idx] == self.debug:

            printf("Stress rate")
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx], d_sigma_dot[idx + 1],
                   d_sigma_dot[idx + 2])
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx + 3],
                   d_sigma_dot[idx + 4], d_sigma_dot[idx + 5])
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx + 6],
                   d_sigma_dot[idx + 7], d_sigma_dot[idx + 8])
            printf("\n")

            printf("Plastic strain rate")
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx], d_eps_p_dot[idx + 1],
                   d_eps_p_dot[idx + 2])
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx + 3],
                   d_eps_p_dot[idx + 4], d_eps_p_dot[idx + 5])
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx + 6],
                   d_eps_p_dot[idx + 7], d_eps_p_dot[idx + 8])
            printf( "====================================")

    def _get_helpers_(self):
        return[yield_criterion, decompose_stress, matrix_multiply,
               matrix_transpose]

# Using the softening model of Anastasopoulos et al. J. Geotech. Geoenviron. Eng., 2007, 133(8): 943-958
# Note this is only for plane-strain, self.sim_dim == 2:
class DruckerPragerSolverWSoft(Equation):

    def __init__(self, dest, sources, y_criterion, c_model=1, debug=0):
        self.c_model = c_model
        self.y_criterion = y_criterion
        self.debug = debug
        super(DruckerPragerSolverWSoft, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_q, d_sigma, d_eps_p, d_sigma_tr, d_sigma_dev,
                   d_sigma_dot, d_eps_dot, d_spin_dot, d_eps_p_dot, d_ep_acc, d_phi_p, d_phi_res, d_psi_p,d_sy, d_eps_p_oct,d_eps_p_f,d_phi, d_psi, d_psi_res,
                   d_eta, d_h_mod, d_bulk, d_shear, d_flag, d_gid, dt):

        i, idx = declare("int", 2)
        p, q = declare("matrix(1)", 2)
        sigma, sig_dot, sig_spin, sig_spin_t = declare("matrix(9)", 4)
        spin_dot = declare("matrix(9)")

        # Index used to access values in arrays
        idx = int(9*d_idx)

        # Flag particle to use trial values
        d_flag[d_idx] = 0

        # Calculate octahedral plastic shear strain
        I1= d_eps_p[idx] + d_eps_p[idx + 4] + d_eps_p[idx + 8]
        e2 = 0.0
        for i in range(9):
            e = d_eps_p[idx + i]
            if i % 4 == 0:
                e -= I1/3
            e2 += e*e
        q_eps = sqrt(3.0*e2 / 2.0)
        eps_p_oct=q_eps*2*sqrt(2)/3
        d_eps_p_oct[d_idx]=eps_p_oct


        # Calculate mobilized friction and dilatancy angles
        phi_p=d_phi_p[d_idx]#*180/pi
        phi_res=d_phi_res[d_idx]#*180/pi
        psi_p=d_psi_p[d_idx]#*180/pi
        psi_res=d_psi_res[d_idx]#*180/pi
        eps_p_f=d_eps_p_f[d_idx]


        if eps_p_oct>=0 and  eps_p_f>eps_p_oct:
            phi=phi_p-(phi_p-phi_res)*(eps_p_oct/eps_p_f)
            psi=psi_p*(1-(eps_p_oct/eps_p_f))
        else:
            phi=phi_res
            psi=psi_res

        d_phi[d_idx]=phi*180/pi
        d_psi[d_idx]=psi*180/pi
        
        # Calculate D-P parameters
        sy = d_sy[d_idx]
        aphi = sqrt(6)*tan(phi) / \
               sqrt(3 + 4*pow(tan(phi), 2))
        apsi = sqrt(6)*tan(psi) / \
               sqrt(3 + 4*pow(tan(psi), 2))
        ac = sqrt(2)/sqrt(3 + 4*pow(tan(phi), 2))

        # Initialize stress and stress rate tensors
        for i in range(9):
            sigma[i] = d_sigma_tr[idx + i]
            sig_dot[i] = d_sigma_dot[idx + i]

        # If not elastic material
        if self.c_model > 0:

            p[0] = d_p[d_idx]
            q[0] = d_q[d_idx]
            h_mod = 0.0

            # Initialize temporary vectors
            for i in range(9):
                sigma[i] = d_sigma[idx + i]
                spin_dot[i] = d_spin_dot[idx + i]

            # Check for yielding
            y = yield_criterion(q, p, aphi, ac, sy, self.y_criterion)

            # If yielding
            if y > 1e-6:

                # Elastic constants
                k = d_bulk[d_idx]
                g = d_shear[d_idx]

                # Plastic modulus
                if sy > 0:  # No hardening if no cohesion
                    h_mod = d_h_mod[d_idx]

                # Plastic multiplier
                dgamma = y / (2*g + k*aphi*apsi + ac*ac*h_mod)

                # Volumetric strain rate
                eps_dot_v = d_eps_dot[idx] + d_eps_dot[idx + 4] + \
                            d_eps_dot[idx + 8]

                # Check if return to the cone is valid
                norm_s = sqrt(2.0/3.0)*q[0]
                if norm_s - 2*g*dgamma < 0.0:

                    # Initialize variables for apex calculations
                    eps_e_dot_v = 0.0
                    dgamma = 0.0
                    pn = (sigma[0] + sigma[4] + sigma[8]) / 3.0

                    if apsi != 0.0:

                        # Plastic multiplier and rate
                        dgamma = (aphi*p[0] - ac*sy) / \
                                 (k*aphi*apsi + ac*ac*h_mod)
                        dg_dot = dgamma / dt

                        # Volumetric plastic strain rate
                        eps_p_dot_v = dg_dot*apsi

                        # Volumetric and deviatoric elastic strain rate
                        eps_e_dot_v = eps_dot_v - eps_p_dot_v

                        # Update stress and strain rates
                        for i in range(9):

                            # Get rid of spin rate
                            spin_dot[i] = 0.0

                            # Update plastic strain rate
                            ep_dot = d_eps_dot[idx + i]
                            if i % 4 == 0:
                                ep_dot = eps_p_dot_v / 3.0
                            d_eps_p_dot[idx + i] = ep_dot

                            # Updated small strain stress
                            dp = k*eps_e_dot_v*dt
                            s_i = -sigma[i]
                            if i % 4 == 0:
                                s_i += pn + dp
                            sigma[i] += s_i

                            # Updated stress rate
                            sig_dot[i] = s_i/dt

                    else:
                        # If dilation angle is zero, the particle is treated as
                        #  if all excess deformation is plastic and the maximum
                        #  stress is: p_max = ac*sy/aphi
                        for i in range(9):

                            p_i = 0.0
                            p_max = ac * sy / aphi  # Maximum tensile stress
                            eps_e_dot = 0.0
                            eps_e_dot_v = (p_max - pn) / (k * dt)
                            if i % 4 == 0:
                                p_i = p_max
                                eps_e_dot = eps_e_dot_v / 3.0

                            sig_dot[i] = (p_i - sigma[i]) / dt
                            spin_dot[i] = 0.0
                            d_eps_p_dot[idx + i] = d_eps_dot[idx + i] - \
                                                   eps_e_dot

                # Return to the smooth part of the cone
                else:

                    # Rate of plastic multiplier
                    dg_dot = dgamma / dt

                    for i in range(9):

                        # Deviatoric plastic flow directions
                        n_i = d_sigma_dev[idx + i] / norm_s

                        # Total strain rate decomposition
                        eps_dot_d = d_eps_dot[idx + i]
                        if i % 4 == 0:
                            eps_dot_d -= eps_dot_v/3.0

                        # Plastic strain rate
                        eps_p_dot_d = dg_dot*n_i
                        eps_p_dot_v = dg_dot*apsi
                        eps_p_dot_h = 0.0
                        if i % 4 == 0:
                            eps_p_dot_h = eps_p_dot_v / 3.0
                        d_eps_p_dot[idx + i] = eps_p_dot_d + eps_p_dot_h

                        # Stress rate
                        eps_e_dot_d = eps_dot_d - eps_p_dot_d
                        eps_e_dot_v = eps_dot_v - eps_p_dot_v
                        p_i = 0.0
                        if i % 4 == 0:
                            p_i = k*eps_e_dot_v
                        sdot = p_i + 2*g*eps_e_dot_d

                        # Small-strain stress rate
                        sig_dot[i] = sdot

                        # Small-strain updated stress
                        sigma[i] += sdot*dt

                # Update accumulated plastic strain and uniaxial yield stress
                dlamb = dgamma*ac
                d_ep_acc[d_idx] += dlamb
                d_sy[d_idx] += h_mod*dlamb

                # Flag particle to update stress and strain
                d_flag[d_idx] = 1

                if self.debug and d_gid[d_idx] == self.debug:
                    printf("\n")
                    printf("====== Called DPSolver ======")
                    printf('p trial')
                    printf("%.16f\n", d_p[d_idx])
                    printf('\n')
                    printf('q trial')
                    printf("%.16f\n", d_q[d_idx])
                    printf('\n')

                    printf('Sy_n+1,tr')
                    printf("%.16f\n", sy)
                    printf('\n')
                    printf('Sy_n+1')
                    printf("%.16f\n", d_sy[d_idx])
                    printf('\n')
                    printf('aphi, apsi, ac, ac*sy, ac*sy/aphi')
                    printf("%.6f %.6f %.6f %.6f %.9f\n", aphi, apsi, ac, ac*sy,
                           ac*sy/aphi)
                    printf('\n')

                    printf('Y_tr')
                    printf("%.6f\n", y)
                    printf('\n')

        # Jaumann stress rate (~ large deformation)
        matrix_multiply(sigma, spin_dot, sig_spin, 3)
        matrix_multiply(spin_dot, sigma, sig_spin_t, 3)

        for i in range(9):
            d_sigma_dot[idx + i] = sig_dot[i] - sig_spin[i] + \
                                   sig_spin_t[i]

        # Update stress invariants
        # TODO: Find out how to do this in the output rather than every time
        #   step here!!!
        p_dot = (d_sigma_dot[idx] + d_sigma_dot[idx + 4] +
                 d_sigma_dot[idx + 8]) / 3.0
        p_n = (d_sigma[idx] + d_sigma[idx + 4] + d_sigma[idx + 8]) / 3.0
        ph = p_n + p_dot*dt
        d_p[d_idx] = ph

        norm_s2 = 0.0
        for i in range(9):
            s_i = d_sigma[idx + i] + d_sigma_dot[idx + i]*dt
            if i % 4 == 0:
                s_i -= ph
            norm_s2 += s_i * s_i
        d_q[d_idx] = sqrt(3*norm_s2 / 2)

        if self.debug and d_gid[d_idx] == self.debug:

            printf("Stress rate")
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx], d_sigma_dot[idx + 1],
                   d_sigma_dot[idx + 2])
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx + 3],
                   d_sigma_dot[idx + 4], d_sigma_dot[idx + 5])
            printf("%.6f %.6f %.6f\n", d_sigma_dot[idx + 6],
                   d_sigma_dot[idx + 7], d_sigma_dot[idx + 8])
            printf("\n")

            printf("Plastic strain rate")
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx], d_eps_p_dot[idx + 1],
                   d_eps_p_dot[idx + 2])
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx + 3],
                   d_eps_p_dot[idx + 4], d_eps_p_dot[idx + 5])
            printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx + 6],
                   d_eps_p_dot[idx + 7], d_eps_p_dot[idx + 8])
            printf("====================================")

    def _get_helpers_(self):
        return[yield_criterion, decompose_stress, matrix_multiply,
               matrix_transpose]

class ViscousRegularization(Equation):

    def __init__(self, dest, sources, c_model=1, y_criterion=2, tolerance=1e-9,
                 debug=0):
        self.c_model = c_model
        self.y_criterion = y_criterion
        self.tol = tolerance
        self.debug = debug
        super(ViscousRegularization, self).__init__(dest, sources)

    def initialize(self, d_idx, d_sigma, d_sigma_tr, d_eps_e, d_eps_p, d_eta,
                   d_bulk, d_shear, d_gid, dt):

        # TODO: IMPLEMENT THIS LATER (11/24/2019)

        if self.c_model == 2:
            theta = dt / d_eta[d_idx]
            sig_i = 0.0
            tr_sig = 0.0

            for i in range(9):
                sig_i = (d_sigma_tr[9*d_idx + i] +
                         theta*d_sigma[9*d_idx + i]) / (1 + theta)
                d_eps[3*d_idx + i] = eps_i
                d_eps_dev[3*d_idx + i] = \
                    (eps_dev0[i] + theta * d_eps_dev[3*d_idx + i]) / \
                    (1 + theta)
                tr_sig += eps_i 



"""
This class is a multi-model. 
Based on the value under the label "model" in the input file,
assign each layer a different constitutive model

"""
class MultiModel(Equation):
    # def __init__(self, nu=0.3, c_model=1, y_criterion=3, tol=1e-5, max_iter=100, debug=0, *args, **kwargs):
    def __init__(self, dest, sources, nu, c_model, tol, max_iter, debug):
        self.nu = nu
        self.c_model = c_model
        self.tol = tol
        self.max_iter = max_iter
        self.debug = debug
        # super().__init__(dest, sources)
        super(MultiModel, self).__init__(dest, sources)


    def initialize(self, d_model, d_idx, d_p, d_q, d_sigma, d_eps_p, d_sigma_tr, d_sigma_dev,d_eps_p_norm,
                   d_sigma_dot, d_eps_dot, d_spin_dot, d_eps_p_dot, d_ep_acc, d_phi_p, d_phi_res, d_psi_p, d_sy, d_eps_p_oct, d_eps_p_f, d_phi, d_psi, d_psi_res,
                   d_eta, d_h_mod, d_bulk, d_shear, d_flag, d_gid, dt, d_void_ratio, d_void_ref, d_pc, d_lambda_mcc, d_kappa_mcc, d_ms):
        i, idx, count = declare("int", 3)
        sigma, sig_dot, sig_spin, sig_spin_t = declare("matrix(9)", 4)
        spin_dot = declare("matrix(9)")
        n = declare("matrix(9)")
    
        if d_model[d_idx] == 2:
            
            # Index used to access values in arrays
            idx = int(9*d_idx)

            # Flag particle to use trial values
            d_flag[d_idx] = 0

            # Calculate octahedral plastic shear strain
            I1= d_eps_p[idx] + d_eps_p[idx + 4] + d_eps_p[idx + 8]
            e2 = 0.0
            for i in range(9):
                e = d_eps_p[idx + i]
                if i % 4 == 0:
                    e -= I1/3
                e2 += e*e
            q_eps = sqrt(3.0*e2 / 2.0)
            eps_p_oct=q_eps*2*sqrt(2)/3
            d_eps_p_oct[d_idx]=eps_p_oct


            # Calculate mobilized friction and dilatancy angles
            phi_p=d_phi_p[d_idx]#*180/pi
            phi_res=d_phi_res[d_idx]#*180/pi
            psi_p=d_psi_p[d_idx]#*180/pi
            psi_res=d_psi_res[d_idx]#*180/pi
            eps_p_f=d_eps_p_f[d_idx]


            if eps_p_oct>=0 and  eps_p_f>eps_p_oct:
                phi=phi_p-(phi_p-phi_res)*(eps_p_oct/eps_p_f)
                psi=psi_p*(1-(eps_p_oct/eps_p_f))
            else:
                phi=phi_res
                psi=psi_res

            d_phi[d_idx]=phi*180/pi
            d_psi[d_idx]=psi*180/pi
            
            # Calculate D-P parameters
            sy = d_sy[d_idx]
            aphi = sqrt(6)*tan(phi) / \
                sqrt(3 + 4*pow(tan(phi), 2))
            apsi = sqrt(6)*tan(psi) / \
                sqrt(3 + 4*pow(tan(psi), 2))
            ac = sqrt(2)/sqrt(3 + 4*pow(tan(phi), 2))

            # Initialize stress and stress rate tensors
            for i in range(9):
                sigma[i] = d_sigma_tr[idx + i]
                sig_dot[i] = d_sigma_dot[idx + i]

            # If not elastic material
            if self.c_model > 0:

                p = d_p[d_idx]
                q = d_q[d_idx]
                h_mod = 0.0

                # Initialize temporary vectors
                for i in range(9):
                    sigma[i] = d_sigma[idx + i]
                    spin_dot[i] = d_spin_dot[idx + i]

                # Check for yielding
                y = yield_criterion2(q, p, aphi, ac, sy)

                # If yielding
                if y > 1e-6:

                    # Elastic constants
                    k = d_bulk[d_idx]
                    g = d_shear[d_idx]

                    # Plastic modulus
                    if sy > 0:  # No hardening if no cohesion
                        h_mod = d_h_mod[d_idx]

                    # Plastic multiplier
                    dgamma = y / (2*g + k*aphi*apsi + ac*ac*h_mod)

                    # Volumetric strain rate
                    eps_dot_v = d_eps_dot[idx] + d_eps_dot[idx + 4] + \
                                d_eps_dot[idx + 8]

                    # Check if return to the cone is valid
                    norm_s = sqrt(2.0/3.0)*q
                    if norm_s - 2*g*dgamma < 0.0:

                        # Initialize variables for apex calculations
                        eps_e_dot_v = 0.0
                        dgamma = 0.0
                        pn = (sigma[0] + sigma[4] + sigma[8]) / 3.0

                        if apsi != 0.0:

                            # Plastic multiplier and rate
                            dgamma = (aphi*p - ac*sy) / \
                                    (k*aphi*apsi + ac*ac*h_mod)
                            dg_dot = dgamma / dt

                            # Volumetric plastic strain rate
                            eps_p_dot_v = dg_dot*apsi

                            # Volumetric and deviatoric elastic strain rate
                            eps_e_dot_v = eps_dot_v - eps_p_dot_v

                            # Update stress and strain rates
                            for i in range(9):

                                # Get rid of spin rate
                                spin_dot[i] = 0.0

                                # Update plastic strain rate
                                ep_dot = d_eps_dot[idx + i]
                                if i % 4 == 0:
                                    ep_dot = eps_p_dot_v / 3.0
                                d_eps_p_dot[idx + i] = ep_dot

                                # Updated small strain stress
                                dp = k*eps_e_dot_v*dt
                                s_i = -sigma[i]
                                if i % 4 == 0:
                                    s_i += pn + dp
                                sigma[i] += s_i

                                # Updated stress rate
                                sig_dot[i] = s_i/dt

                        else:
                            # If dilation angle is zero, the particle is treated as
                            #  if all excess deformation is plastic and the maximum
                            #  stress is: p_max = ac*sy/aphi
                            for i in range(9):

                                p_i = 0.0
                                p_max = ac * sy / aphi  # Maximum tensile stress
                                eps_e_dot = 0.0
                                eps_e_dot_v = (p_max - pn) / (k * dt)
                                if i % 4 == 0:
                                    p_i = p_max
                                    eps_e_dot = eps_e_dot_v / 3.0

                                sig_dot[i] = (p_i - sigma[i]) / dt
                                spin_dot[i] = 0.0
                                d_eps_p_dot[idx + i] = d_eps_dot[idx + i] - \
                                                    eps_e_dot

                    # Return to the smooth part of the cone
                    else:

                        # Rate of plastic multiplier
                        dg_dot = dgamma / dt

                        for i in range(9):

                            # Deviatoric plastic flow directions
                            n_i = d_sigma_dev[idx + i] / norm_s

                            # Total strain rate decomposition
                            eps_dot_d = d_eps_dot[idx + i]
                            if i % 4 == 0:
                                eps_dot_d -= eps_dot_v/3.0

                            # Plastic strain rate
                            eps_p_dot_d = dg_dot*n_i
                            eps_p_dot_v = dg_dot*apsi
                            eps_p_dot_h = 0.0
                            if i % 4 == 0:
                                eps_p_dot_h = eps_p_dot_v / 3.0
                            d_eps_p_dot[idx + i] = eps_p_dot_d + eps_p_dot_h

                            # Stress rate
                            eps_e_dot_d = eps_dot_d - eps_p_dot_d
                            eps_e_dot_v = eps_dot_v - eps_p_dot_v
                            p_i = 0.0
                            if i % 4 == 0:
                                p_i = k*eps_e_dot_v
                            sdot = p_i + 2*g*eps_e_dot_d

                            # Small-strain stress rate
                            sig_dot[i] = sdot

                            # Small-strain updated stress
                            sigma[i] += sdot*dt

                    # Update accumulated plastic strain and uniaxial yield stress
                    dlamb = dgamma*ac
                    d_ep_acc[d_idx] += dlamb
                    d_sy[d_idx] += h_mod*dlamb

                    # Flag particle to update stress and strain
                    d_flag[d_idx] = 1

                    if self.debug and d_gid[d_idx] == self.debug:
                        printf("\n")
                        printf("====== Called DPSolver ======")
                        printf('p trial')
                        printf("%.16f\n", d_p[d_idx])
                        printf('\n')
                        printf('q trial')
                        printf("%.16f\n", d_q[d_idx])
                        printf('\n')

                        printf('Sy_n+1,tr')
                        printf("%.16f\n", sy)
                        printf('\n')
                        printf('Sy_n+1')
                        printf("%.16f\n", d_sy[d_idx])
                        printf('\n')
                        printf('aphi, apsi, ac, ac*sy, ac*sy/aphi')
                        printf("%.6f %.6f %.6f %.6f %.9f\n", aphi, apsi, ac, ac*sy,
                            ac*sy/aphi)
                        printf('\n')

                        printf('Y_tr')
                        printf("%.6f\n", y)
                        printf('\n')

            # Jaumann stress rate (~ large deformation)
            matrix_multiply(sigma, spin_dot, sig_spin, 3)
            matrix_multiply(spin_dot, sigma, sig_spin_t, 3)

            for i in range(9):
                d_sigma_dot[idx + i] = sig_dot[i] - sig_spin[i] + \
                                    sig_spin_t[i]

            # Update stress invariants
            # TODO: Find out how to do this in the output rather than every time
            #   step here!!!
            p_dot = (d_sigma_dot[idx] + d_sigma_dot[idx + 4] +
                    d_sigma_dot[idx + 8]) / 3.0
            p_n = (d_sigma[idx] + d_sigma[idx + 4] + d_sigma[idx + 8]) / 3.0
            ph = p_n + p_dot*dt
            d_p[d_idx] = ph

            norm_s2 = 0.0
            for i in range(9):
                s_i = d_sigma[idx + i] + d_sigma_dot[idx + i]*dt
                if i % 4 == 0:
                    s_i -= ph
                norm_s2 += s_i * s_i
            d_q[d_idx] = sqrt(3*norm_s2 / 2)

            if self.debug and d_gid[d_idx] == self.debug:

                printf("Stress rate")
                printf("%.6f %.6f %.6f\n", d_sigma_dot[idx], d_sigma_dot[idx + 1],
                    d_sigma_dot[idx + 2])
                printf("%.6f %.6f %.6f\n", d_sigma_dot[idx + 3],
                    d_sigma_dot[idx + 4], d_sigma_dot[idx + 5])
                printf("%.6f %.6f %.6f\n", d_sigma_dot[idx + 6],
                    d_sigma_dot[idx + 7], d_sigma_dot[idx + 8])
                printf("\n")

                printf("Plastic strain rate")
                printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx], d_eps_p_dot[idx + 1],
                    d_eps_p_dot[idx + 2])
                printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx + 3],
                    d_eps_p_dot[idx + 4], d_eps_p_dot[idx + 5])
                printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx + 6],
                    d_eps_p_dot[idx + 7], d_eps_p_dot[idx + 8])
                printf("====================================")
        if d_model[d_idx] == 3:
            
            # Index used to access values in arrays
            idx = int(9 * d_idx)

            # Flag particle to use trial values
            d_flag[d_idx] = 0

            # Consolidation parameters
            lambda_mcc = d_lambda_mcc[d_idx]
            kappa_mcc = d_kappa_mcc[d_idx]
            void_ratio = d_void_ratio[d_idx]
            void_ref = d_void_ref[d_idx]
            v= (1+void_ratio)/(lambda_mcc-kappa_mcc)
            pc0 = -98000

            # Initialize stress and stress rate tensors
            for i in range(9):
                sigma[i] = d_sigma_tr[idx + i]
                sig_dot[i] = d_sigma_dot[idx + i]

            # If not elastic material
            if self.c_model > 0:
                p = d_p[d_idx]
                q = d_q[d_idx]
                pc = d_pc[d_idx]
                pcn = pc
                ms = d_ms[d_idx]

                # Initialize temporary vectors
                for i in range(9):
                    sigma[i] = d_sigma[idx + i]
                    spin_dot[i] = d_spin_dot[idx + i]

                # Check for yielding
                y = yield_criterion_mcc(q, p, pc, ms)

                # If yielding
                if y > 1e-8:

                    # Trial hydrostatic stress and von Mises stress
                    ptr = p
                    qtr = q

                    # Elastic constants
                    k = d_bulk[d_idx]
                    g = d_shear[d_idx]
                    

                    # Counter of iterations for NR
                    count = 0

                    # Initialize consistency parameter
                    dg = 0.0

                    # Newton-Raphson loop
                    while y > self.tol and count < self.max_iter:

                        # Pre-calculate the partial derivatives of the yield surface w.r.t. the consistency parameter, dg (DeltaL_k in ConsUpdate.m Enrique Approximate CPP)
                        a = (-2*k * pow(dg * (2*p - pc) + 1/v, 2) * (p - pc/2)) / \
                            (8*k * pow(p - pc/2, 2) * pow(dg, 3) + 8*(k * (1/v) + p/2 - pc/4) * (p - pc/2) *
                            pow(dg, 2) + 2*(1/v) * (k * (1/v) + 2*p - pc - pcn/2) * dg + pow(1/v, 2))

                        b1 = -q / (dg + pow(ms, 2) / (6*g))

                        c1 = ((-2*p + pc) * (1/v) * pcn) / \
                            (8*k * pow(p - pc/2, 2) * pow(dg, 3) + 8*(k * (1/v) + p/2 - pc/4) * (p - pc/2) *
                            pow(dg, 2) + 2*(1/v) * (k * (1/v) + 2*p - pc - pcn/2) * dg + pow(1/v, 2))

                        # Derivative of the yield function (Fp_k)
                        dy = (2*p - pc) * a + (2*q / pow(ms, 2)) * b1 - p * c1

                        # Update the consistency parameter
                        dg -= y/dy

                        # Auxiliary values to find pc
                        b2 = (1/v) * (1 + 2*k * dg) + 2*dg * ptr
                        c2 = pcn * (1/v) * (1 + 2*k * dg)
                        d1 = (b2 / dg)
                        d2 = sqrt(pow(b2 / dg, 2) - (4*c2) / dg)

                        # Roots of second order polynomial used to determine pc
                        pc_a = .5*(d1 + d2)
                        pc_b = .5*(d1 - d2)

                        # Update pc
                        if pc_a < 0:
                            pc = pc_a
                        else:
                            pc = pc_b

                        # Update p and q
                        p = (ptr + k*dg * pc) / (1 + 2*k * dg)
                        q = qtr / (1 + 6*g * dg / pow(ms, 2))

                        # Calculate if stress on the yield surface
                        y = yield_criterion_mcc(q, p, pc, ms)

                        # Increment iteration counter
                        count += 1

                    # Check if convergence obtained
                    if y > self.tol:
                        printf("\n")
                        printf("====== NR DID NOT CONVERGE AFTER ")
                        printf("%d",count)
                        printf("ITERATIONS ======")
                        printf('Residual F')
                        printf("%.16f\n", y)
                        printf('\n')

                    d_pc[d_idx] = pc

                    # Update bulk and shear moduli
                    #if p != 0:
                    #    void_ratio = void_ref - lambda_mcc * log(-pc/1000) + kappa_mcc * log(pc/p)
                    #    k = abs((1+void_ratio)*p/kappa_mcc)
                    #    g = 3*k * (1 - 2*self.nu) / (2*(1 + self.nu))
                    #    d_bulk[d_idx] = k
                    #    d_void_ratio[d_idx] = void_ratio
                    #    d_shear[d_idx] = g

                    # Rate of plastic multiplier
                    dg_dot = dg / dt

                    # Volumetric strain rate
                    eps_dot_v = d_eps_dot[idx] + d_eps_dot[idx + 4] + \
                                d_eps_dot[idx + 8]

                    # Construct deformation tensors
                    norm_s = sqrt(2.0 / 3.0) * qtr
                    for i in range(9):

                        if norm_s > 0:
                            # Deviatoric plastic flow directions
                            n_i = d_sigma_dev[idx + i] / norm_s
                        else:
                            n_i=0
                        
                        # d_n[idx+i] = n_i

                        # Total strain rate decomposition
                        eps_dot_d = d_eps_dot[idx + i]
                        if i % 4 == 0:
                            eps_dot_d -= eps_dot_v / 3.0

                        # Plastic strain rates
                        eps_p_dot_d = sqrt(6) * dg_dot * q * n_i / (ms * ms)
                        eps_p_dot_v = dg_dot * (2*p - pc)
                        eps_p_dot_h = 0.0
                        if i % 4 == 0:
                            eps_p_dot_h = eps_p_dot_v / 3.0
                        d_eps_p_dot[idx + i] = eps_p_dot_d + eps_p_dot_h

                        # Stress rate
                        eps_e_dot_d = eps_dot_d - eps_p_dot_d
                        eps_e_dot_v = eps_dot_v - eps_p_dot_v
                        p_i = 0.0
                        if i % 4 == 0:
                            p_i = k * eps_e_dot_v
                        sdot = p_i + 2 * g * eps_e_dot_d

                        # Small-strain stress rate
                        sig_dot[i] = sdot

                        # Small-strain updated stress
                        sigma[i] += sdot * dt

                    # Update accumulated plastic strain and uniaxial yield stress
                    d_ep_acc[d_idx] += dg

                    # Flag particle to update stress and strain
                    d_flag[d_idx] = 1

                    if self.debug and d_gid[d_idx] == self.debug:
                        printf("\n")
                        printf("====== Called DPSolver ======")
                        printf('p trial')
                        printf("%.16f\n", d_p[d_idx])
                        printf('\n')
                        printf('q trial')
                        printf("%.16f\n", d_q[d_idx])
                        printf('\n')

            # Jaumann stress rate (~ large deformation)
            matrix_multiply(sigma, spin_dot, sig_spin, 3)
            matrix_multiply(spin_dot, sigma, sig_spin_t, 3)

            for i in range(9):
                d_sigma_dot[idx + i] = sig_dot[i] - sig_spin[i] + \
                                    sig_spin_t[i]

            # Update stress invariants
            # TODO: Find out how to do this in the output rather than every time
            #   step here!!!
            p_dot = (d_sigma_dot[idx] + d_sigma_dot[idx + 4] +
                    d_sigma_dot[idx + 8]) / 3.0
            p_n = (d_sigma[idx] + d_sigma[idx + 4] + d_sigma[idx + 8]) / 3.0
            ph = p_n + p_dot * dt
            d_p[d_idx] = ph
            p=d_p[d_idx]

            norm_s2 = 0.0
            for i in range(9):
                s_i = d_sigma[idx + i] + d_sigma_dot[idx + i] * dt
                if i % 4 == 0:
                    s_i -= ph
                norm_s2 += s_i * s_i
            d_q[d_idx] = sqrt(3 * norm_s2 / 2)

            # Update bulk and shear moduli
            if p != 0:
                ph=ph/1000
                pc=pc/1000
                void_ratio = void_ref - lambda_mcc * log(-pc/1000) + kappa_mcc * log(pc/ph)
                ph=ph*1000
                k = abs((1+void_ratio)*p/kappa_mcc)
                g = 3*k * (1 - 2*self.nu) / (2*(1 + self.nu))
                d_bulk[d_idx] = k
                d_void_ratio[d_idx] = void_ratio
                d_shear[d_idx] = g

            #if d_idx==182:
            #    print(log(-p))
            #    print(void_ratio)

            if self.debug and d_gid[d_idx] == self.debug:
                printf("Stress rate")
                printf("%.6f %.6f %.6f\n", d_sigma_dot[idx], d_sigma_dot[idx + 1],
                    d_sigma_dot[idx + 2])
                printf("%.6f %.6f %.6f\n", d_sigma_dot[idx + 3],
                    d_sigma_dot[idx + 4], d_sigma_dot[idx + 5])
                printf("%.6f %.6f %.6f\n", d_sigma_dot[idx + 6],
                    d_sigma_dot[idx + 7], d_sigma_dot[idx + 8])
                printf("\n")

                printf("Plastic strain rate")
                printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx], d_eps_p_dot[idx + 1],
                    d_eps_p_dot[idx + 2])
                printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx + 3],
                    d_eps_p_dot[idx + 4], d_eps_p_dot[idx + 5])
                printf("%.6f %.6f %.6f\n", d_eps_p_dot[idx + 6],
                    d_eps_p_dot[idx + 7], d_eps_p_dot[idx + 8])
                printf( "====================================")
        
        summ=0.0
        for i in range(9):
            summ += (d_eps_p_dot[i+idx]*dt)*(d_eps_p_dot[i+idx]*dt)
        d_eps_p_norm[d_idx] += summ
        #printf(d_eps_p[d_idx])
        
    def _get_helpers_(self):
        return [yield_criterion_mcc, matrix_multiply, yield_criterion2, decompose_stress, matrix_transpose]
        