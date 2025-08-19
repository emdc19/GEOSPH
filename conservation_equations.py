from pysph.sph.equation import Equation
from compyle.api import declare
from math import pow


class MomentumEquation(Equation):

    def __init__(self, dest, sources, sim_dim=2, sigma_c=0.0, debug=0):
        self.sim_dim = sim_dim
        self.debug = debug
        self.sigma_c=sigma_c
        super(MomentumEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_gid, d_rho, d_sigma, d_au, d_av, d_aw, s_idx,
             s_gid, s_m, s_rho, s_sigma, DWIJ,d_fi):
        i, j = declare("int", 2)
        dvdt = declare("matrix(3)")
        fi=d_fi[d_idx]

        for i in range(3):
            dvdt[i] = 0.0

        m = s_m[s_idx]
        rhoi2 = pow(d_rho[d_idx], 2)
        rhoj2 = pow(s_rho[s_idx], 2)

        for i in range(3):
            for j in range(3):
                sij = d_sigma[9*d_idx + 3*i + j] / rhoi2 + \
                     s_sigma[9*s_idx + 3*i + j] / rhoj2
                #sij=(d_sigma[9*d_idx + 3*i + j]+s_sigma[9*s_idx + 3*i + j])/(d_rho[d_idx] * s_rho[s_idx])
                dvdt[i] += m*sij*DWIJ[j]

        #if fi<200: # 0.7 for 3D .55 for 2D as per Bui
        if self.sigma_c != 0.0:
        #if fi<.998:
            vj= m/(d_rho[d_idx] * s_rho[s_idx])
            s_c = -2 * vj * self.sigma_c
            dvdt[0] += s_c * DWIJ[0]
            dvdt[1] += s_c * DWIJ[1]
            dvdt[2] += s_c * DWIJ[2]

        d_au[d_idx] += dvdt[0]
        d_av[d_idx] += dvdt[1]
        d_aw[d_idx] += dvdt[2]

        if d_gid[d_idx] == self.debug:
            printf("\n")
            printf("====== Called MomentumEquation ======")
            printf("Neighbors")
            printf("%d\n", s_gid[s_idx])
            printf("%d\n", d_gid[d_idx])
            printf("\n")
            printf("DW")
            printf("%.9f %.9f %.9f\n", DWIJ[0], DWIJ[1], DWIJ[2])
            printf("\n")
            printf("Sigma_d")
            printf("%.9f %.9f %.9f\n", d_sigma[9*d_idx], d_sigma[9*d_idx + 1],
                   d_sigma[9*d_idx + 2])
            printf("%.9f %.9f %.9f\n", d_sigma[9*d_idx + 3],
                   d_sigma[9*d_idx + 4],
                   d_sigma[9*d_idx + 5])
            printf("%.9f %.9f %.9f\n", d_sigma[9*d_idx + 6],
                   d_sigma[9*d_idx + 7],
                   d_sigma[9*d_idx + 8])
            printf("\n")
            printf("Sigma_s")
            printf("%.9f %.9f %.9f\n", s_sigma[9 * s_idx],
                   s_sigma[9 * s_idx + 1],
                   s_sigma[9 * s_idx + 2])
            printf("%.9f %.9f %.9f\n", s_sigma[9 * s_idx + 3],
                   s_sigma[9 * s_idx + 4],
                   s_sigma[9 * s_idx + 5])
            printf("%.9f %.9f %.9f\n", s_sigma[9 * s_idx + 6],
                   s_sigma[9 * s_idx + 7],
                   s_sigma[9 * s_idx + 8])
            printf("====================================")

    def post_loop(self, d_idx, d_rho, d_u, d_v, d_w, d_au, d_av, d_aw, d_se,
                  d_te, d_gid, d_as_a):
        idx = declare("int")

        # Add artificial stress acceleration
        idx = 3*d_idx
        d_au[d_idx] += d_as_a[idx]
        d_av[d_idx] += d_as_a[idx + 1]
        d_aw[d_idx] += d_as_a[idx + 2]

        # Update total energy
        d_te[d_idx] += d_se[d_idx]

        if self.debug and d_gid[d_idx] == self.debug:
            printf("\n")
            printf("====== Final accelerations ======")
            printf("%.9f %.9f %.9f\n", d_au[d_idx], d_av[d_idx], d_aw[d_idx])
            printf("====================================")


class DensityEquation(Equation):
    def __init__(self, dest, sources, debug=0):
        self.debug = debug
        super(DensityEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_gid, d_arho, s_gid, s_idx, s_m, VIJ, DWIJ):
        div_vij = VIJ[0]*DWIJ[0] + VIJ[1]*DWIJ[1] + VIJ[2]*DWIJ[2]
        d_arho[d_idx] += s_m[s_idx]*div_vij

        if d_gid[d_idx] == self.debug:
            printf("\n")
            printf("====== Called Density Update ======")
            printf("Neighbors")
            printf("%d\n", s_gid[s_idx])
            printf("%d\n", d_gid[d_idx])
            printf("\n")
            printf("DW")
            printf("%.9f %.9f %.9f\n", DWIJ[0], DWIJ[1], DWIJ[2])
            printf("\n")
            printf("vij")
            printf("%.9f %.9f %.9f\n", VIJ[0], VIJ[1], VIJ[2])
            printf("\n")
            printf("d_rho")
            printf("%.9f\n", d_arho[d_idx])
            printf("====================================")

    def post_loop(self, d_idx, d_gid, d_arho):
        if self.debug and d_gid[d_idx] == self.debug:
            printf("\n")
            printf("====== Final Density ======")
            printf("Particle")
            printf("%d\n", d_gid[d_idx])
            printf("\n")
            printf("Density rate")
            printf("%.6f\n", d_arho[d_idx])
            printf("====================================")


class SummationDensity(Equation):
    def __init__(self, dest, sources, freq=50, debug=0):
        self.freq = freq
        self.debug = debug
        super(SummationDensity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_gid, d_rho, d_wsum, s_idx, s_gid, s_m, WIJ, t, dt):
        f, n = declare("int", 2)

        # Summation density calculation frequency and step number
        f = int(self.freq)
        n = int(t / dt)

        if self.freq > 0 and n % f == 0:
            d_rho[d_idx] += s_m[s_idx]*WIJ / d_wsum[d_idx]

        if d_gid[d_idx] == self.debug:
            printf("\n")
            printf("====== Called Density Update ======")
            printf("Neighbors")
            printf("%d\n", s_gid[s_idx])
            printf("%d\n", d_gid[d_idx])
            printf("\n")
            printf("Wij")
            printf("%.9f\n", WIJ)
            printf("\n")
            printf("d_rho")
            printf("%.9f\n", d_rho[d_idx])
            printf("====================================")

    def post_loop(self, d_idx, d_gid, d_rho):
        if self.debug and d_gid[d_idx] == self.debug:
            printf("\n")
            printf("====== Final Density ======")
            printf("Particle")
            printf("%d\n", d_gid[d_idx])
            printf("\n")
            printf("Density rate")
            printf("%.6f\n", d_rho[d_idx])
            printf("====================================")
