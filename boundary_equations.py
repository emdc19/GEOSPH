from pysph.sph.equation import Equation
from compyle.api import declare


class BoundaryStress(Equation):
    def __init__(self, dest, sources, sim_dim=2, debug_bound=0):
        self.dim = sim_dim
        self.debug_bound = debug_bound
        super(BoundaryStress, self).__init__(dest, sources)

    def initialize(self, d_idx, d_sigma):
        i = declare("int")
        for i in range(9):
            d_sigma[9*d_idx + i] = 0.0

    def loop(self, d_idx, d_gid, d_sigma, d_wsum, s_idx, s_gid, s_m, s_rho,
             s_sigma, s_f, XIJ, WIJ):
        i = declare("int")

        # Extrapolate stress to the boundary particle
        m = s_m[s_idx]
        rho = s_rho[s_idx]
        wsum = d_wsum[d_idx]
        for i in range(9):
            d_sigma[9*d_idx + i] += m*s_sigma[9*s_idx + i]*WIJ / (rho*wsum)

        if self.debug_bound and d_gid[d_idx] == self.debug_bound:
            printf("\n")
            printf("====== Called Boundary Stress ======")
            printf("Neighbors")
            printf("%d\n", s_gid[s_idx])
            printf("%d\n", d_gid[d_idx])
            printf("\n")
            printf("W")
            printf("%.9f \n", WIJ)
            printf("\n")
            printf("Sigma_s")
            printf("%.9f %.9f %.9f\n", s_sigma[9*s_idx], s_sigma[9*s_idx + 1],
                   s_sigma[9*s_idx + 2])
            printf("%.9f %.9f %.9f\n", s_sigma[9*s_idx + 3],
                   s_sigma[9*s_idx + 4],
                   s_sigma[9*s_idx + 5])
            printf("%.9f %.9f %.9f\n", s_sigma[9*s_idx + 6],
                   s_sigma[9*s_idx + 7],
                   s_sigma[9*s_idx + 8])
            printf("====================================")

    def post_loop(self, d_idx, d_sigma, d_gid):
        if self.debug_bound and d_gid[d_idx] == self.debug_bound:
            printf("\n")
            printf("====== Final Boundary Stress ======")
            printf('Boundary stress - Sigma')
            printf("%.16e %.16e %.16e\n", d_sigma[9*d_idx],
                   d_sigma[9*d_idx + 1], d_sigma[9*d_idx + 2])
            printf("%.16e %.16e %.16e\n", d_sigma[9*d_idx + 3],
                   d_sigma[9*d_idx + 4], d_sigma[9*d_idx + 5])
            printf("%.16e %.16e %.16e\n", d_sigma[9*d_idx + 6],
                   d_sigma[9*d_idx + 7], d_sigma[9*d_idx + 8])
            printf("====================================")


class ConfiningBCtreatment(Equation):
    def __init__(self, dest, sources, debug_bound=0):
        self.debug_bound = debug_bound
        super(ConfiningBCtreatment, self).__init__(dest, sources)

    def initialize(self, d_idx, d_fi):
        d_fi[d_idx] = 0.0

    def loop(self, d_idx, d_gid, d_rho, d_sigma, d_au, d_av, d_aw, s_idx,
     s_gid, s_m, s_rho, s_sigma, DWIJ,WIJ,d_fi):
        m = s_m[s_idx]
        rhoj= s_rho[s_idx]
        fi=(m/rhoj)*WIJ
        d_fi[d_idx]+=fi


class FrictionalBCParameters(Equation):
    def __init__(self, dest, sources, debug_bound=0):
        self.debug_bound = debug_bound
        super(FrictionalBCParameters, self).__init__(dest, sources)

    def initialize(self, d_idx, d_n_bar, d_v_bar):
        i, idx = declare("int", 2)
        idx = 3*d_idx
        for i in range(3):
            d_n_bar[idx] = 0.0
            d_v_bar[idx] = 0.0
            idx += 1

    def loop(self, d_idx, d_n_bar, d_v_bar, d_wsum, s_idx, s_m, s_rho, s_u,
             s_v, s_w, s_n, WIJ):
        i, idx, jdx = declare("int", 3)

        # Calculate some constants
        idx = 3*d_idx
        jdx = 3*s_idx
        w_bar = s_m[s_idx]*WIJ / (s_rho[s_idx]*d_wsum[d_idx])

        # Average boundary unit normal vector
        for i in range(3):
            d_n_bar[idx + i] += s_n[jdx + i]*w_bar

        # Average boundary velocity
        d_v_bar[idx] += s_u[jdx]*w_bar
        d_v_bar[idx + 1] += s_v[jdx]*w_bar
        d_v_bar[idx + 2] += s_w[jdx]*w_bar


class FrictionalBCForce(Equation):
    def __init__(self, dest, sources, debug_bound=0):
        self.debug_bound = debug_bound
        super(FrictionalBCForce, self).__init__(dest, sources)

    def initialize(self, d_idx, d_u, d_v, d_w, d_v_bar, d_n_bar, d_af):
        i, idx = declare("int", 2)
        v, v_n, v_t = declare("matrix(3)", 3)
        idx = 3*d_idx

        # Initialize the frictional force and velocity vectors
        for i in range(3):
            d_af[idx + i] = 0.0
            v[i] = d_v_bar[idx + i]
            v_n[i] = 0.0
            v_t[i] = 0.0

        # Calculate the relative velocity between soil and boundary
        v[0] -= d_u[idx]
        v[1] -= d_v[idx]
        v[2] -= d_w[idx]

        # Calculate the normal relative velocity
        vn = v[0]*d_n_bar[idx] + v[1]*d_n_bar[idx + 1] + v[2]*d_n_bar[idx + 2]
        for i in range(3):
            v_n[i] = vn*d_v_bar[idx + i]

        # Calculate the tangential relative velocity
        for i in range(3):
            v_t[i] = v[i] - v_n[i]

        vt = sqrt(v_t[0]*v_t[0] + v_t[1]*v_t[1] + v_t[1]*v_t[1])
