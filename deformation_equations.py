from pysph.sph.equation import Equation
from compyle.api import declare
from pysph.sph.wc.density_correction import gj_solve

"""
This class ...
"""


class DeformationRates(Equation):
    def __init__(self, dest, sources, debug=0):
        self.debug = debug
        super(DeformationRates, self).__init__(dest, sources)

    def initialize(self, d_idx, d_eps_dot, d_spin_dot):
        i = declare("int")

        for i in range(9):
            d_eps_dot[9*d_idx + i] = 0.0
            d_spin_dot[9 * d_idx + i] = 0.0

    def loop(self, d_idx, d_gid, d_eps_dot, d_spin_dot, s_idx, s_gid, s_m,
             s_rho, VIJ, DWIJ):

        i, j = declare("int", 2)
        grad_v = declare("matrix(9)")

        # Initialize grad_v
        for i in range(9):
            grad_v[i] = 0.0

        # Calculate the velocity gradient
        for i in range(3):
            dvij = s_m[s_idx] * -VIJ[i] / s_rho[s_idx]
            for j in range(3):
                grad_v[3*i + j] += dvij * DWIJ[j]

        # Calculate strain rate and spin rate tensors
        for i in range(3):
            for j in range(3):
                gv_ij = grad_v[3*i + j]
                gv_ji = grad_v[3*j + i]
                d_eps_dot[9*d_idx + 3*i + j] += 0.5*(gv_ij + gv_ji)
                d_spin_dot[9*d_idx + 3*i + j] += 0.5*(gv_ij - gv_ji)

        if d_gid[d_idx] == self.debug and self.debug:
            printf("\n")
            printf("====== Called DeformationRates ======")
            printf("Neighbors")
            printf("%d %d\n", d_gid[d_idx], s_gid[s_idx])
            printf("\n")

            printf('Relative particle velocity')
            printf("%.9f %.9f %.9f\n", VIJ[0], VIJ[1], VIJ[2])
            printf("\n")

            printf("rhoj")
            printf("%.6f\n", s_rho[s_idx])
            printf("\n")

            printf('DWIJ')
            printf("%.9f %.9f %.9f\n", DWIJ[0], DWIJ[1], DWIJ[2])
            printf("\n")

            printf('Gradient of v')
            printf("%.16e %.16e %.16e\n", grad_v[0], grad_v[1], grad_v[2])
            printf("%.16e %.16e %.16e\n", grad_v[3], grad_v[4], grad_v[5])
            printf("%.16e %.16e %.16e\n", grad_v[6], grad_v[7], grad_v[8])
            printf('\n')

            printf('Strain Rate, eps_dot')
            printf("%.16e %.16e %.16e\n", d_eps_dot[9*d_idx],
                   d_eps_dot[9*d_idx + 1], d_eps_dot[9*d_idx + 2])
            printf("%.16e %.16e %.16e\n", d_eps_dot[9*d_idx + 3],
                   d_eps_dot[9*d_idx + 4], d_eps_dot[9*d_idx + 5])
            printf("%.16e %.16e %.16e\n", d_eps_dot[9*d_idx + 6],
                   d_eps_dot[9*d_idx + 7], d_eps_dot[9*d_idx + 8])
            printf('\n')

            printf('Spin Rate, omega_dot')
            printf("%.16e %.16e %.16e\n", d_spin_dot[9*d_idx],
                   d_spin_dot[9*d_idx + 1], d_spin_dot[9*d_idx + 2])
            printf("%.16e %.16e %.16e\n", d_spin_dot[9*d_idx + 3],
                   d_spin_dot[9*d_idx + 4], d_spin_dot[9*d_idx + 5])
            printf("%.16e %.16e %.16e\n", d_spin_dot[9*d_idx + 6],
                   d_spin_dot[9*d_idx + 7], d_spin_dot[9*d_idx + 8])
            printf("=========================")


class FiniteDifferenceDeformation(Equation):
    def __init__(self, dest, sources, debug=0):
        self.debug = debug
        super(FiniteDifferenceDeformation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_eps_dot, d_spin_dot):
        i = declare("int")

        for i in range(9):
            d_eps_dot[9*d_idx + i] = 0.0
            d_spin_dot[9 * d_idx + i] = 0.0

    def loop(self, d_idx, d_gid, d_eps_dot, d_spin_dot, d_wsum, s_idx, s_gid,
             s_m, s_rho, XIJ, R2IJ, VIJ, WIJ, EPS):

        i, j = declare("int", 2)
        grad_v = declare("matrix(9)")

        # Initialize grad_v
        for i in range(9):
            grad_v[i] = 0.0

        # Calculate the velocity gradient
        wij = WIJ / d_wsum[d_idx]
        for i in range(3):
            avij = 2*s_m[s_idx] * VIJ[i]*wij / \
                   (s_rho[s_idx]*R2IJ + EPS)
            for j in range(3):
                grad_v[3*i + j] += avij*XIJ[j]

        # Calculate strain rate and spin rate tensors
        for i in range(3):
            for j in range(3):
                gv_ij = grad_v[3*i + j]
                gv_ji = grad_v[3*j + i]
                d_eps_dot[9*d_idx + 3*i + j] += 0.5*(gv_ij + gv_ji)
                d_spin_dot[9*d_idx + 3*i + j] += 0.5*(gv_ij - gv_ji)

        if d_gid[d_idx] == self.debug and self.debug:
            printf("\n")
            printf("====== Called DeformationRates ======")
            printf("Neighbors")
            printf("%d %d\n", d_gid[d_idx], s_gid[s_idx])
            printf("\n")

            printf('Relative particle velocity')
            printf("%.9f %.9f %.9f\n", VIJ[0], VIJ[1], VIJ[2])
            printf("\n")

            printf("rhoj")
            printf("%.6f\n", s_rho[s_idx])
            printf("\n")

            printf('WIJ')
            printf("%.9f\n", WIJ)
            printf("\n")

            printf('Gradient of v')
            printf("%.16e %.16e %.16e\n", grad_v[0], grad_v[1], grad_v[2])
            printf("%.16e %.16e %.16e\n", grad_v[3], grad_v[4], grad_v[5])
            printf("%.16e %.16e %.16e\n", grad_v[6], grad_v[7], grad_v[8])
            printf('\n')

            printf('Strain Rate, eps_dot')
            printf("%.16e %.16e %.16e\n", d_eps_dot[9*d_idx],
                   d_eps_dot[9*d_idx + 1], d_eps_dot[9*d_idx + 2])
            printf("%.16e %.16e %.16e\n", d_eps_dot[9*d_idx + 3],
                   d_eps_dot[9*d_idx + 4], d_eps_dot[9*d_idx + 5])
            printf("%.16e %.16e %.16e\n", d_eps_dot[9*d_idx + 6],
                   d_eps_dot[9*d_idx + 7], d_eps_dot[9*d_idx + 8])
            printf('\n')

            printf('Spin Rate, omega_dot')
            printf("%.16e %.16e %.16e\n", d_spin_dot[9*d_idx],
                   d_spin_dot[9*d_idx + 1], d_spin_dot[9*d_idx + 2])
            printf("%.16e %.16e %.16e\n", d_spin_dot[9*d_idx + 3],
                   d_spin_dot[9*d_idx + 4], d_spin_dot[9*d_idx + 5])
            printf("%.16e %.16e %.16e\n", d_spin_dot[9*d_idx + 6],
                   d_spin_dot[9*d_idx + 7], d_spin_dot[9*d_idx + 8])
            printf("=========================")
