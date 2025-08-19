from pysph.sph.equation import Equation
from compyle.api import declare
from pysph.sph.wc.density_correction import gj_solve
from math import sqrt


class KernelSum(Equation):
    def __init__(self, dest, sources, debug=0, bp=0):
        self.debug = debug
        self.bound_part = bp
        super(KernelSum, self).__init__(dest, sources)

    def initialize(self, d_idx, d_wsum):
        d_wsum[d_idx] = 0.0

    def loop(self, d_idx, d_wsum, s_idx, s_m, s_rho, WIJ):
        d_wsum[d_idx] += s_m[s_idx]*WIJ / s_rho[s_idx]

    def post_loop(self, d_idx, d_wsum, d_gid):
        if d_wsum[d_idx] == 0.0:
            d_wsum[d_idx] = 1.0

        if self.debug and d_gid[d_idx] == self.bound_part:
            printf("\n")
            printf('Boundary kernel sum')
            printf("%.6f\n", d_wsum[d_idx])
            printf("\n")


class KernelCorrection(Equation):
    def __init__(self, dest, sources):
        super(KernelCorrection, self).__init__(dest, sources)

    def loop(self, d_idx, d_wsum, WIJ):
        WIJ /= d_wsum[d_idx]


class GradientCorrectionPreStep(Equation):
    def __init__(self, dest, sources, kgc=0, sim_dim=2, debug=0):
        self.calc = kgc
        self.dim = sim_dim
        self.debug = debug
        super(GradientCorrectionPreStep, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m_mat):
        i = declare('int')

        if self.calc:
            for i in range(9):
                d_m_mat[9*d_idx + i] = 0.0

    def loop_all(self, d_idx, d_gid, d_m_mat, d_x, d_y, d_z, d_h, s_m, s_rho,
                 s_x, s_y, s_z, s_h, SPH_KERNEL, NBRS, N_NBRS):
        i, j, k, s_idx, n = declare("int", 4)
        xij, dwij = declare("matrix(3)", 2)

        if self.calc:
            x = d_x[d_idx]
            y = d_y[d_idx]
            z = d_z[d_idx]
            h = d_h[d_idx]
            n = self.dim

            for k in range(N_NBRS):
                s_idx = NBRS[k]
                xij[0] = x - s_x[s_idx]
                xij[1] = y - s_y[s_idx]
                xij[2] = z - s_z[s_idx]
                hij = 0.5*(d_h[d_idx] + s_h[s_idx])
                r = sqrt(xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2])
                SPH_KERNEL.gradient(xij, r, hij, dwij)
                v = s_m[s_idx] / s_rho[s_idx]
                if r > 1.0e-12:
                    for i in range(n):
                        for j in range(n):
                            d_m_mat[9*d_idx + 3*i + j] -= v*dwij[i]*xij[j]

            if self.debug and d_gid[d_idx] == self.debug:
                printf("\n")
                printf('L matrix')
                printf("%.16e %.16e %.16e\n", d_m_mat[9*d_idx],
                       d_m_mat[9*d_idx + 1], d_m_mat[9*d_idx + 2])
                printf("%.16e %.16e %.16e\n", d_m_mat[9*d_idx + 3],
                       d_m_mat[9*d_idx + 4], d_m_mat[9*d_idx + 5])
                printf("%.16e %.16e %.16e\n", d_m_mat[9*d_idx + 6],
                       d_m_mat[9*d_idx + 7], d_m_mat[9*d_idx + 8])
                printf('\n')


class GradientCorrection(Equation):
    r"""**Kernel Gradient Correction**

    From [BonetLok1999], equations (42) and (45)

    .. math::
            \nabla \tilde{W}_{ab} = L_{a}\nabla W_{ab}

    .. math::
            L_{a} = \left(\sum \frac{m_{b}}{\rho_{b}} \nabla W_{ab}
            \mathbf{\otimes}x_{ba} \right)^{-1}
    """

    def __init__(self, dest, sources, kgc=0, sim_dim=2, tol=0.1):
        self.calc = kgc
        self.dim = sim_dim
        self.tol = tol
        super(GradientCorrection, self).__init__(dest, sources)

    def loop(self, d_idx, d_m_mat, DWIJ, HIJ):
        i, j, n, nt = declare('int', 4)
        temp = declare('matrix(12)')
        res = declare('matrix(3)')

        if self.calc:
            n = self.dim
            nt = n + 1

            for i in range(n):
                for j in range(n):
                    temp[nt*i + j] = d_m_mat[9*d_idx + 3*i + j]
                # Augmented part of matrix
                temp[nt*i + n] = DWIJ[i]

            gj_solve(temp, n, 1, res)

            # This is from the original PySPH reference docs. I don't know why
            #  they have these checks of tolerance, but they do not work well.
            # eps = 1.0e-04 * HIJ
            # res_mag = 0.0
            # dwij_mag = 0.0
            # for i in range(n):
            #     res_mag += abs(res[i])
            #     dwij_mag += abs(DWIJ[i])
            # change = abs(res_mag - dwij_mag)/(dwij_mag + eps)
            # if change < self.tol:
            for i in range(n):
                DWIJ[i] = res[i]

    def _get_helpers_(self):
        return [gj_solve]
