from pysph.sph.integrator_step import IntegratorStep
from math import fabs, sqrt, sin
from compyle.api import declare


class SedimentEulerStep(IntegratorStep):

    def __init__(self, gravity=-10.0, damp_time=0.0, sim_dim=2, debug=0):
        self.gravity = gravity
        self.damp_time = damp_time
        self.sim_dim = sim_dim
        self.debug = debug
        super(SedimentEulerStep, self).__init__()

    def initialize(self):
        pass

    def stage1(self, d_idx, d_m, d_rho, d_arho, d_u, d_v, d_w, d_au, d_av,
               d_aw, d_x, d_y, d_z, d_disp, d_eps, d_eps_e, d_eps_p, d_eps_dot,
               d_eps_p_dot, d_sigma, d_sigma_dot, d_p, d_q, d_pe, d_ke, d_te, d_a_ver, d_a_hor, d_seis_time,d_nstep,
               d_gid, d_flag, d_h, d_young, d_f, t, dt):
        r"""
        The damping term:

        .. math::

            -C_d \vec{v}

        where :math:`C_d` is given by:

        .. math::

            C_d = \xi \sqrt \frac{E}{\rho h^2}

        Is taken from the paper of Bui and Fukagawa (2013).

        :param d_idx:
        :param d_m:
        :param d_rho:
        :param d_arho:
        :param d_u:
        :param d_v:
        :param d_w:
        :param d_au:
        :param d_av:
        :param d_aw:
        :param d_x:
        :param d_y:
        :param d_z:
        :param d_disp:
        :param d_eps:
        :param d_eps_e:
        :param d_eps_p:
        :param d_eps_dot:
        :param d_eps_p_dot:
        :param d_sigma:
        :param d_sigma_dot:
        :param d_p:
        :param d_q:
        :param d_pe:
        :param d_ke:
        :param d_te:
        :param d_gid:
        :param d_flag:
        :param d_h:
        :param d_young:
        :param d_f:
        :param t:
        :param dt:
        :return:

        References
        ----------
        .. [BuiFukagawa2013]
        H.H. Bui, R. Fukagawa (2013) "An improved SPH method for saturated
        soils and its application to investigate the mechanisms of embankment
        failure: Case of hydrostatic pore-water pressure." Int. J. Numer.
        Anal. Meth. Geomech. Vol.37, p. 31–50
        """
        i, idx = declare("int", 2)

        # Gravity acceleration
        g = self.gravity

        # Damping parameters
        damp_time = self.damp_time
        cd = 0.0
        xi = 0.05  # Should be between 0.01 and 0.25 for 0.1% and 5% dv

        # Reduce gravity during damping period
        if damp_time > 0.0:
            if t <= damp_time:
                cd = -xi*sqrt(d_young[d_idx] / d_rho[d_idx]) / d_h[d_idx]
            else:
                cd = 0.0

        # Add gravity
        if self.sim_dim == 2:
            d_av[d_idx] += g
        else:
            d_aw[d_idx] += g

        # Accelerogram data
        #time=t
        #timestep=d_seis_time.index(time) # to ensure we add the acceleration at time t
        #timestep= (t/.002)-1
        #int ts= (int) timestep
        #a_hori=d_a_hor
        #timestep=int(timestep)
        #ts=timestep.copy()

        #nstep=d_nstep[d_idx]
        a_h_t = d_a_hor[d_nstep[d_idx]]
        a_v_t = d_a_ver[d_nstep[d_idx]]
        d_nstep[d_idx] +=1

        #d_counter[d_idx]=d_counter[d_idx]+1
        #a_v_t=d_a_ver[timestep]
        

        # Add damping and body forces to acceleration
        idx = 3*d_idx
        d_au[d_idx] += cd*d_u[d_idx] + d_f[idx] #+ a_h_t*(1+2*pow(d_y[d_idx]/52,2))*9.8        #+.5*9.8*sin(50*t)*(1+2*pow(d_y[d_idx]/52,2))
        d_av[d_idx] += cd*d_v[d_idx] + d_f[idx + 1] #+ a_v_t*(1+2*pow(d_y[d_idx]/52,2))*9.8        #+.2**sin(50*t+1)*(1+2*pow(d_y[d_idx]/52,2))
        d_aw[d_idx] += cd*d_w[d_idx] + d_f[idx + 2]

        # Update velocity
        d_u[d_idx] += dt*d_au[d_idx]
        d_v[d_idx] += dt*d_av[d_idx]
        d_w[d_idx] += dt*d_aw[d_idx]
        #if d_x[d_idx]>56:
        #    if d_y[d_idx]<50:
        #        d_v[d_idx]=-.2#*(d_y[d_idx])/50
        #    else:
        #        d_v[d_idx]=.2#*(100-d_y[d_idx])/50

        # Update the values of position and relative displacement
        dx = dt*d_u[d_idx]
        dy = dt*d_v[d_idx]
        dz = dt*d_w[d_idx]

        # Update position
        d_x[d_idx] += dx
        d_y[d_idx] += dy
        d_z[d_idx] += dz

        # Update accumulated displacement
        d_disp[idx] += dx
        d_disp[idx + 1] += dy
        d_disp[idx + 2] += dz

        # Update total, plastic, and elastic strains, and Cauchy stress
        for i in range(9):
            deps = dt*d_eps_dot[9*d_idx + i]
            d_sigma[9*d_idx + i] += dt*d_sigma_dot[9*d_idx + i]
            d_eps[9*d_idx + i] += deps

            if d_flag[d_idx] == 0:
                d_eps_e[9*d_idx + i] += deps
            else:
                deps_p = dt*d_eps_p_dot[9*d_idx + i]
                d_eps_p[9*d_idx + i] += deps_p
                d_eps_e[9*d_idx + i] += deps - deps_p

        # Update density
        d_rho[d_idx] += dt*d_arho[d_idx]

        # Update potential energy
        h = d_y[d_idx]
        i = 1
        if self.sim_dim == 3:
            h = d_z[d_idx]
            i = 2

        d_pe[d_idx] = d_m[d_idx]*fabs(h*d_f[idx+i])

        # Update kinetic energy
        u = d_u[d_idx]
        v = d_v[d_idx]
        w = d_w[d_idx]
        d_ke[d_idx] = 0.5*d_m[d_idx]*(u*u + v*v + w*w)

        # Update total energy
        d_te[d_idx] = d_pe[d_idx] + d_ke[d_idx]

        if self.debug and d_gid[d_idx] == self.debug:
            printf("\n")
            printf("Updated position: %.9f %.9f %.9f\n", d_x[d_idx],
                   d_y[d_idx], d_z[d_idx])
            printf("Updated velocity: %.9f %.9f %.9f\n", d_u[d_idx],
                   d_v[d_idx], d_w[d_idx])
            printf("Updated stress")
            printf("%.9f %.9f %.9f\n", d_sigma[9*d_idx], d_sigma[9*d_idx + 1],
                   d_sigma[9*d_idx + 2])
            printf("%.9f %.9f %.9f\n", d_sigma[9*d_idx + 3],
                   d_sigma[9*d_idx + 4],  d_sigma[9*d_idx + 5])
            printf("%.9f %.9f %.9f\n", d_sigma[9*d_idx + 6],
                   d_sigma[9*d_idx + 7], d_sigma[9*d_idx + 8])
            printf('p')
            printf("%.9f\n", d_p[d_idx])
            printf("\n")
            printf('q')
            printf("%.9f\n", d_q[d_idx])
            printf("\n")
            printf("Updated strain")
            printf("%.9f %.9f %.9f\n", d_eps[9 * d_idx],
                   d_eps[9 * d_idx + 1], d_eps[9 * d_idx + 2])
            printf("%.9f %.9f %.9f\n", d_eps[9 * d_idx + 3],
                   d_eps[9 * d_idx + 4], d_eps[9 * d_idx + 5])
            printf("%.9f %.9f %.9f\n", d_eps[9 * d_idx + 6],
                   d_eps[9 * d_idx + 7], d_eps[9 * d_idx + 8])
            printf("\n")
            printf("Plastic strain")
            printf("%.9f %.9f %.9f\n", d_eps_p[9 * d_idx],
                   d_eps_p[9 * d_idx + 1], d_eps_p[9 * d_idx + 2])
            printf("%.9f %.9f %.9f\n", d_eps_p[9 * d_idx + 3],
                   d_eps_p[9 * d_idx + 4], d_eps_p[9 * d_idx + 5])
            printf("%.9f %.9f %.9f\n", d_eps_p[9 * d_idx + 6],
                   d_eps_p[9 * d_idx + 7], d_eps_p[9 * d_idx + 8])
            printf("\n")
            printf("Elastic strain")
            printf("%.9f %.9f %.9f\n", d_eps_e[9 * d_idx],
                   d_eps_e[9 * d_idx + 1], d_eps_e[9 * d_idx + 2])
            printf("%.9f %.9f %.9f\n", d_eps_e[9 * d_idx + 3],
                   d_eps_e[9 * d_idx + 4], d_eps_e[9 * d_idx + 5])
            printf("%.9f %.9f %.9f\n", d_eps_e[9 * d_idx + 6],
                   d_eps_e[9 * d_idx + 7], d_eps_e[9 * d_idx + 8])
            printf("\n")

    def stage2(self):
        pass


class BoundaryEulerStep(IntegratorStep):

    def __init__(self, debug=0, bp=0):
        self.debug = debug
        self.bound_part = bp
        super(BoundaryEulerStep, self).__init__()

    def initialize(self):
        pass

    def stage1(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w, d_du, d_f, d_gid,
               dt):

        # Update velocity with prescribed body forces (accelerations)
        d_u[d_idx] += dt*d_f[3*d_idx]
        d_v[d_idx] += dt*d_f[3*d_idx + 1]
        d_w[d_idx] += dt*d_f[3*d_idx + 2]

        # Update position
        d_x[d_idx] += dt*d_u[d_idx]
        d_y[d_idx] += dt*d_v[d_idx]
        d_z[d_idx] += dt*d_w[d_idx]

        if self.debug and d_gid[d_idx] == self.bound_part:
            printf("\n")
            printf("New boundary position: %.9f %.9f %.9f\n", d_x[d_idx],
                   d_y[d_idx], d_z[d_idx])
            printf("\n")

    def stage2(self):
        pass
