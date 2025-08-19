import cython
from textwrap import dedent
from pysph.sph.equation import Equation
from compyle.api import declare
from math import floor


class MonaghanArtificialViscosity(Equation):
    r"""
    Classical Monaghan artificial viscosity [Monaghan, 2005]_

    .. math::

        \frac{d\mathbf{v}_a}{dt} = -\sum_{b}m_{b}\Pi_{ab}\nabla_{a}W_{ab}

    where

    .. math::

        \Pi_{ab}=\begin{cases}\frac{-\alpha_{\pi}\bar{c}_{ab}\phi_{ab}+
        \beta_{\pi}\phi_{ab}^{2}}{\bar{\rho}_{ab}}, & \mathbf{v}_{ab}\cdot
        \mathbf{r}_{ab}<0\\0, & \mathbf{v}_{ab}\cdot\mathbf{r}_{ab}\geq0
        \end{cases}

    with

    .. math::

        \phi_{ab}=\frac{h\mathbf{v}_{ab}\cdot\mathbf{r}_{ab}}
        {|\mathbf{r}_{ab}|^{2}+\epsilon^{2}}\\

        \bar{c}_{ab}&=&\frac{c_{a}+c_{b}}{2}\\

        \bar{\rho}_{ab}&=&\frac{\rho_{a}+\rho_{b}}{2}

    References
    ----------
    .. [Monaghan2005] J. Monaghan, "Smoothed particle hydrodynamics",
        Reports on Progress in Physics, 68 (2005), pp. 1703-1759.
    """

    def __init__(self, dest, sources, alpha=0.5, beta=0.2, debug=0):
        r"""
        Parameters
        ----------
        alpha : float
            produces a shear and bulk viscosity
        beta : float
            used to handle high Mach number shocks
        """
        self.alpha = alpha
        self.beta = beta
        self.debug = debug
        super(MonaghanArtificialViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_au, d_av, d_aw, d_cs, d_gid, s_idx, s_m, s_cs,
             s_gid, VIJ, XIJ, HIJ, R2IJ, RHOIJ1, EPS, DWIJ):

        if self.alpha != 0.0 or self.beta != 0.0:

            # Tests if particles are approaching or departing from each other.
            v_rel = VIJ[0]*XIJ[0] + VIJ[1]*XIJ[1] + VIJ[2]*XIJ[2]

            if v_rel < 0:

                # Average sound speed velocity
                cij = 0.5*(d_cs[d_idx] + s_cs[s_idx])

                muij = (HIJ*v_rel) / (R2IJ + EPS)
                piij = (self.alpha*cij*muij - self.beta*muij*muij)*RHOIJ1
                m_p = s_m[s_idx]*piij

                d_au[d_idx] += m_p*DWIJ[0]
                d_av[d_idx] += m_p*DWIJ[1]
                d_aw[d_idx] += m_p*DWIJ[2]

                if self.debug and d_gid[d_idx] == self.debug:
                    printf("\n")
                    printf("====== Called ArtVisc ======")
                    printf("part i, part j: %d %d\n", d_gid[d_idx],
                           s_gid[s_idx])
                    printf("rij^2 + eps: %.6f\n", R2IJ + EPS)
                    printf("h: %.6f\n", HIJ)
                    printf("v_rel: %.6f\n", v_rel)
                    printf("muij: %.6f\n", muij)
                    printf("piij: %.6f\n", piij)
                    printf("m_p: %.6f\n", s_m[s_idx] * piij)
                    printf("rho_ij: %.6f\n", 1/RHOIJ1)
                    printf("alpha: %.6f\n", self.alpha)
                    printf("beta: %.6f\n", self.beta)
                    printf("c0: %.6f\n", cij)
                    printf("DW")
                    printf("%.9f %.9f %.9f\n", DWIJ[0], DWIJ[1], DWIJ[2])
                    printf("====================================")


    def post_loop(self, d_idx, d_au, d_av, d_aw, d_gid):
        if self.debug and d_gid[d_idx] == self.debug and (self.alpha != 0.0 or
                                                          self.beta != 0.0):
            printf("\n")
            printf("=== Acceleration with artificial viscosity ===")
            printf("%.9f %.9f %.9f\n", d_au[d_idx], d_av[d_idx], d_aw[d_idx])
            printf("====================================")


class XSPHCorrection(Equation):
    r"""
    Position stepping with XSPH correction [Monaghan1992]

    .. math::

        \frac{d\mathbf{r}_{a}}{dt}=\mathbf{\hat{v}}_{a}=\mathbf{v}_{a}-
        \epsilon\sum_{b}m_{b}\frac{\mathbf{v}_{ab}}{\bar{\rho}_{ab}}W_{ab}

    References
    ----------
    .. [Monaghan1992] J. Monaghan, Smoothed Particle Hydrodynamics, "Annual
        Review of Astronomy and Astrophysics", 30 (1992), pp. 543-574.
    """

    def __init__(self, dest, sources, eps=0.5, freq=30, debug=0):
        r"""
        Parameters
        ----------
        eps : float
            :math:`\epsilon` as in the above equation

        Notes
        -----
        This equation must be used to advect the particles. XSPH can be
        turned off by setting the parameter ``eps = 0``.
        """
        self.eps = eps
        self.freq = freq
        self.debug = debug
        super(XSPHCorrection, self).__init__(dest, sources)

    def loop(self, d_idx, d_x, d_y, d_z, d_wsum, s_idx, s_m, WIJ, RHOIJ1, VIJ,
             dt, t, d_gid, s_gid):
        f, n = declare("int", 2)

        # Regularization frequency and step number
        f = int(self.freq)
        n = int(t / dt)

        if self.freq > 0 and n % f == 0 and self.eps > 0.0:
            coef = -self.eps*s_m[s_idx]*WIJ*RHOIJ1 / d_wsum[d_idx]

            d_x[d_idx] += coef*VIJ[0]*dt
            d_y[d_idx] += coef*VIJ[1]*dt
            d_z[d_idx] += coef*VIJ[2]*dt

    def post_loop(self, d_idx, d_x, d_y, d_z, d_gid):
        if self.debug and d_gid[d_idx] == self.debug and self.eps > 0.0:
            printf("\n")
            printf("=== New velocities after XSPH correction ===")
            printf("%.9f %.9f %.9f\n", d_x[d_idx], d_y[d_idx], d_z[d_idx])
            printf("====================================")


class SimplifiedArtificialStress(Equation):
    r"""
    **Simplified artificial stress to remove tensile instability**

    This formulation is a simplified version of the artificial stress proposed
    by Monaghan and was introduced in the work of Peng at al. (2019)

    In this formulation, instead of using the principal stresses, it uses the
    hydrostatic stress instead.

    The added artificial stress term to the balance of linear momentum is given
    by:

    .. math::

        S_{ij} = f^n_{ij} \left(R_i + R_j \right),

    where the components of :math:`R` are given :

    .. math::

        R_i = \frac{\epsilon |p_i|}{\rho^2_i} \text{  if  } p_i < 0.

    References
    ----------
    .. [Peng2019]
    C. Peng, S. Wang, W. Wu, HS. Yu,
    C. Wang, JY. Chen, "LOQUAT: an open-source GPU-accelerated SPH solver for
    geotechnical modeling", "Acta Geotechnica",
    https://doi.org/10.1007/s11440-019-00839-1.
    """
    def __init__(self, dest, sources, as_eps=0.5, dp=0.1, c_model=0, debug=0):
        r"""
        Parameters
        ----------
        eps : float
            constant
        """
        self.dp = dp
        self.c_model = c_model
        self.debug = debug

        # Scaling factor: Keep it equal to or larger than 0.6. Smaller
        #  values do not work for some reason.
        self.eps = as_eps
        eps_min = 0.6
        if as_eps < eps_min:
            self.eps = eps_min

        super(SimplifiedArtificialStress, self).__init__(dest, sources)

    def initialize(self, d_idx, d_as_a, d_wdp, d_h, SPH_KERNEL):
        i = declare("int")
        xij = declare("matrix(3)")

        for i in range(3):
            d_as_a[3*d_idx + i] = 0.0
            xij[i] = 0.0

        # Calculate kernel at dp (wdp)
        xij[0] = self.dp
        d_wdp[d_idx] = SPH_KERNEL.kernel(xij, self.dp, d_h[d_idx])

    def loop(self, d_idx, d_rho, d_p, d_as_a, d_sy, d_wdp, s_idx, s_m, s_rho,
             s_p, WIJ, DWIJ):

        i = declare("int")

        # Check if material has tensile strength and artificial stress active
        if self.eps > 0.0 and (d_sy[d_idx] > 0.0 or self.c_model == 0):

            # Scaling factor
            d = pow(WIJ / d_wdp[d_idx], 4.0)

            # Compute artificial stress pressures
            pi = d_p[d_idx]
            r_i = 0.0
            if pi > 0:
                r_i = pi / (d_rho[d_idx]*d_rho[d_idx])

            pj = s_p[s_idx]
            r_j = 0.0
            if pj > 0.0:
                r_j = pj / (s_rho[s_idx]*s_rho[s_idx])

            # Add artificial stress contribution to the accelerations
            r_ij = -d*self.eps*(r_i + r_j)
            mrij = s_m[s_idx]*r_ij
            for i in range(3):
                d_as_a[3*d_idx + i] += mrij*DWIJ[i]

    def post_loop(self, d_idx, d_as_a, d_gid):
        if self.debug and d_gid[d_idx] == self.debug and self.eps > 0.0:
            printf("\n")
            printf("=== Artificial Stress Acceleration ===")
            printf("%.9f %.9f %.9f\n", d_as_a[3*d_idx], d_as_a[3*d_idx + 1],
                   d_as_a[3*d_idx + 2])
            printf("====================================")


class PySPHArtificialStress(Equation):
    r"""
    **Artificial stress to remove tensile instability**

    The dispersion relations in [Gray2001] are used to determine the
    different components of :math:`R`.
    Angle of rotation for particle :math:`a`

    .. math::
        \tan{2 \theta_a} = \frac{2\sigma_a^{xy}}{\sigma_a^{xx} - \sigma_a^{yy}}

    In rotated frame, the new components of the stress tensor are

    .. math::
        \bar{\sigma}_a^{xx} = \cos^2{\theta_a} \sigma_a^{xx} + 2\sin{\theta_a}
        \cos{\theta_a}\sigma_a^{xy} + \sin^2{\theta_a}\sigma_a^{yy}\\
        \bar{\sigma}_a^{yy} = \sin^2{\theta_a} \sigma_a^{xx} + 2\sin{\theta_a}
        \cos{\theta_a}\sigma_a^{xy} + \cos^2{\theta_a}\sigma_a^{yy}

    Components of :math:`R` in rotated frame:

    .. math::
        \bar{R}_{a}^{xx}=\begin{cases}-\epsilon\frac{\bar{\sigma}_{a}^{xx}}
        {\rho^{2}} & \bar{\sigma}_{a}^{xx}>0\\0 & \bar{\sigma}_{a}^{xx}\leq0
        \end{cases}\\
        \bar{R}_{a}^{yy}=\begin{cases}-\epsilon\frac{\bar{\sigma}_{a}^{yy}}
        {\rho^{2}} & \bar{\sigma}_{a}^{yy}>0\\0 & \bar{\sigma}_{a}^{yy}\leq0
        \end{cases}

    Components of :math:`R` in original frame:

    .. math::
        R_a^{xx} = \cos^2{\theta_a} \bar{R}_a^{xx} +
        \sin^2{\theta_a} \bar{R}_a^{yy}\\
        R_a^{yy} = \sin^2{\theta_a} \bar{R}_a^{xx} +
        \cos^2{\theta_a} \bar{R}_a^{yy}\\
        R_a^{xy} = \sin{\theta_a} \cos{\theta_a}\left(\bar{R}_a^{xx} -
        \bar{R}_a^{yy}\right)
    """

    def __init__(self, dest, sources, as_eps=0.2, dp=0.1, c_model=0, debug=0):
        r"""
        Parameters
        ----------
        eps : float
            constant
        """
        self.eps = as_eps
        self.dp = dp
        self.c_model = c_model
        self.debug = debug
        super(PySPHArtificialStress, self).__init__(dest, sources)

    def _cython_code_(self):
        code = dedent(
            """
            cimport cython
            from pysph.base.linalg3 cimport eigen_decomposition
            from pysph.base.linalg3 cimport transform_diag_inv
            """
        )
        return code

    def initialize(self, d_idx, d_rho, d_p, d_sigma, d_h, d_as_a, d_wdp,
                   d_asig, d_gid, SPH_KERNEL):
        r"""

        :param d_idx:
        :param d_rho:
        :param d_p:
        :param d_sigma:
        :param d_h:
        :param d_as_a:
        :param d_wdp:
        :param d_asig:
        :param d_gid:
        :param SPH_KERNEL:
        :return:
        """
        i, j, idx = declare("int", 3)
        xij = declare("matrix(3)")
        r = declare('matrix((3,3))')  # Matrix of Eigenvectors (columns)
        rab = declare('matrix((3,3))')  # Artificial stress
        s = declare('matrix((3,3))')  # Stress tensor with pressure.
        v = declare('matrix((3,))')  # Eigenvalues
        rd = declare('matrix((3,))')  # Artificial stress principle directions

        # Initialize artificial stress and acceleration tensors
        idx = 9*d_idx
        for i in range(3):
            d_as_a[3*d_idx + i] = 0.0
            for j in range(3):
                d_asig[idx + 3*i + j] = 0.0

        # Check if material has tensile strength and artificial stress active
        if self.eps > 0.0 or self.c_model == 0:

            # initialize variables
            for i in range(3):
                xij[i] = 0.0

            # Calculate kernel at dp (wdp) and the multiplier exponent (n)
            xij[0] = self.dp
            d_wdp[d_idx] = SPH_KERNEL.kernel(xij, self.dp, d_h[d_idx])

            # 1/rho^2
            rho = d_rho[d_idx]
            rho21 = 1.0 / (rho*rho)

            # Initialize the temporary Cauchy stress tensor
            for i in range(3):
                for j in range(3):
                    s[i][j] = d_sigma[idx + 3*i + j]

            # compute the principle stresses
            eigen_decomposition(s, r, cython.address(v[0]))

            # artificial stress corrections
            for i in range(3):
                if v[i] > 0:
                    rd[i] = -self.eps*v[i]*rho21
                else:
                    rd[i] = 0

            # transform artificial stresses in original frame
            transform_diag_inv(cython.address(rd[0]), r, rab)

            # store the values
            for i in range(3):
                for j in range(3):
                    d_asig[idx + 3*i + j] = rab[i][j]

            if self.debug and d_gid[d_idx] == self.debug and self.eps > 0.0:
                printf("\n")
                printf("=== Artificial Stress Acceleration ===")
                printf("%.9f %.9f %.9f\n", d_as_a[3*d_idx],
                       d_as_a[3*d_idx + 1],
                       d_as_a[3*d_idx + 2])
                printf("====================================")

    def loop(self, d_idx, d_asig, d_as_a, s_idx, s_m, s_asig, d_wdp, WIJ,
             DWIJ):
        r"""

        :param d_idx:
        :param d_asig:
        :param d_as_a:
        :param s_idx:
        :param s_m:
        :param s_asig:
        :param d_wdp:
        :param WIJ:
        :param DWIJ:
        :return:
        """
        i, j, idx, isx = declare("int", 4)
        dvdt = declare("matrix(3)")

        # TODO: Check this modification (and others) when using DP model
        if self.eps > 0.0 or self.c_model == 0:

            # initialize variables
            for i in range(3):
                dvdt[i] = 0.0

            # Calculate scaling factor, d = fij^n
            d = pow(WIJ / d_wdp[d_idx], 4.0)

            # Add artificial stress contribution to the accelerations
            idx = 9*d_idx
            isx = 9*s_idx
            for i in range(3):
                for j in range(3):
                    aij = d_asig[idx + 3*i + j] + s_asig[isx + 3*i + j]
                    dvdt[i] += aij*DWIJ[j]

            dm = d*s_m[s_idx]
            for i in range(3):
                d_as_a[3*d_idx + i] += dm*dvdt[i]
