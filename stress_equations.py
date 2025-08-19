from pysph.sph.equation import Equation
from compyle.api import declare
from math import sqrt


class TrialStress(Equation):
    def __init__(self, dest, sources, debug=0):
        self.debug = debug
        super(TrialStress, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gid, d_sigma, d_sigma_tr, d_sigma_dot,
                   d_eps_dot, d_bulk, d_shear, dt):

        i = declare("int")
        deps, deps_dev = declare("matrix(9)", 2)

        # Calculate strain increment
        deps_vol = 0.0
        for i in range(9):
            eps_dot = d_eps_dot[9*d_idx + i]
            deps[i] = eps_dot
            deps_dev[i] = eps_dot
            if i % 4 == 0:
                deps_vol += eps_dot

        deps_vol /= 3.0
        deps_dev[0] -= deps_vol
        deps_dev[4] -= deps_vol
        deps_dev[8] -= deps_vol

        # Material elastic parameters
        k = d_bulk[d_idx]
        g = d_shear[d_idx]

        # Calculate trial stress state
        for i in range(9):
            dsig = 2*g*deps_dev[i]
            if i % 4 == 0:
                dsig += 3.0*k*deps_vol
            d_sigma_tr[9*d_idx + i] = d_sigma[9*d_idx + i] + dsig*dt
            d_sigma_dot[9*d_idx + i] = dsig

        if self.debug and d_gid[d_idx] == self.debug:
            printf("\n")
            printf("====== Called TrialStress ======")
            printf("Sigma trial")
            printf("%.6f %.6f %.6f\n", d_sigma_tr[9*d_idx],
                   d_sigma_tr[9*d_idx + 1], d_sigma_tr[9*d_idx + 2])
            printf("%.6f %.6f %.6f\n", d_sigma_tr[9*d_idx + 3],
                   d_sigma_tr[9*d_idx + 4], d_sigma_tr[9*d_idx + 5])
            printf("%.6f %.6f %.6f\n", d_sigma_tr[9*d_idx + 6],
                   d_sigma_tr[9*d_idx + 7], d_sigma_tr[9*d_idx + 8])
            printf("\n")
            printf("Sigma dot")
            printf("%.6f %.6f %.6f\n", d_sigma_dot[9*d_idx],
                   d_sigma_dot[9*d_idx + 1], d_sigma_dot[9*d_idx + 2])
            printf("%.6f %.6f %.6f\n", d_sigma_dot[9*d_idx + 3],
                   d_sigma_dot[9*d_idx + 4], d_sigma_dot[9*d_idx + 5])
            printf("%.6f %.6f %.6f\n", d_sigma_dot[9*d_idx + 6],
                   d_sigma_dot[9*d_idx + 7], d_sigma_dot[9*d_idx + 8])
            printf("====================================")


class TrialStressDecomposition(Equation):
    def __init__(self, dest, sources, debug=0):
        self.debug = debug
        super(TrialStressDecomposition, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gid, d_sigma_tr, d_p, d_q, d_sigma_dev):
        i = declare("int")

        # Hydrostatic stress
        p = (d_sigma_tr[9*d_idx] + d_sigma_tr[9*d_idx + 4] +
             d_sigma_tr[9*d_idx + 8]) / 3.0

        # Von Mises stress and deviatoric stress tensor
        d_p[d_idx] = p
        s2 = 0.0
        for i in range(9):
            s = d_sigma_tr[9*d_idx + i]
            if i % 4 == 0:
                s -= p
            s2 += s*s
            d_sigma_dev[9*d_idx + i] = s

        d_q[d_idx] = sqrt(3.0*s2 / 2.0)

        if self.debug and d_gid[d_idx] == self.debug:
            printf("\n")
            printf('=== Called StressDecomposition ===')
            printf('p')
            printf("%.6f\n", p)
            printf("\n")
            printf('q')
            printf("%.6f\n", d_q[d_idx])
            printf("====================================")


class StressRegularization(Equation):
    def __init__(self, dest, sources, freq=40, debug=0):
        self.freq = freq
        self.debug = debug
        super(StressRegularization, self).__init__(dest, sources)

    def loop(self, d_idx, d_sigma, d_wsum, s_idx, s_m, s_rho, s_sigma, WIJ, t,
             dt):

        i, idx, isx, f, n = declare("int", 5)

        # Regularization frequency and step number
        f = int(self.freq)
        n = int(t/dt)

        if self.freq > 0 and n % f == 0:
            idx = 9*d_idx
            isx = 9*s_idx
            wsum = d_wsum[d_idx]
            vij = s_m[s_idx]*WIJ / (s_rho[s_idx]*wsum)

            for i in range(9):
                d_sigma[idx + i] += vij*(s_sigma[isx + i] - d_sigma[idx + i])
