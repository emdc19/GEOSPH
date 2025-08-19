import os
import numpy as np
from math import sqrt, pi
from cyarray.carray import IntArray
# =============================================================================
# ================================ PySPH IMPORTS ==============================

# Base imports
from pysph.base.utils import get_particle_array
from pysph.base.kernels import WendlandQuintic, CubicSpline
from pysph.base.nnps import LinkedListNNPS, DomainManager

# Solver and application
from pysph.solver.application import Application
from pysph.solver.solver import Solver

# SPH equation imports
from pysph.sph.equation import Group

# =============================================================================
# ============================ MY OWN MODULES =================================
from io_utils import import_parts_data, import_simulation_parameters, \
    get_csv_header, separate_particle_data_arrays, convert_pysph_output

from integrators import MyEulerIntegrator
from integrator_steppers import SedimentEulerStep, BoundaryEulerStep
from deformation_equations import DeformationRates
from boundary_equations import BoundaryStress, ConfiningBCtreatment
from constitutive_equations import DruckerPragerSolverExact as DPSolver, \
    ModifiedCamClay_Approx_CPP as MCCSolver_Approx_CPP, \
    ModifiedCamClay_Exact_CPP as MCCSolver_Exact_CPP, \
    ModifiedCamClayCRM as MCCSolverCRM, \
    ModifiedCamClay_Exact_Tensile_CPP as MCCSolver_Exact_Tensile_CPP, \
    DamageModel as DModel, \
    ChemicalCoupledDamageModel as DModelC, \
    DruckerPragerSolverWSoft as DPSolverSoft,\
    MultiModel as MultiModel

from conservation_equations_flat import MomentumEquation, DensityEquation, \
    SummationDensity

from stress_equations import TrialStressDecomposition, StressRegularization, \
    TrialStress

from monaghan_equations import MonaghanArtificialViscosity as ArtVisc, \
    PySPHArtificialStress as ArtStress, XSPHCorrection as XSph

from kernel_corrections import GradientCorrection as KernelGradCorrect, \
    GradientCorrectionPreStep as KernelGradLMatrix, KernelSum

# =============================================================================
# ============================ Global variables ===============================
# TXT_PATH = '/mnt/c/Users/alomir/Desktop/Bucknell/2_SCHOLARSHIP/Conferences/' \
#            'DFI_Conference_2021/Simulations/Pile_Driving.txt'
#
# CSV_PATH = '/mnt/c/Users/alomir/Desktop/Bucknell/2_SCHOLARSHIP/Conferences/' \
#            'DFI_Conference_2021/Simulations/Pile_Driving.csv'

# TXT_PATH = '/mnt/c/Users/ahf009/Desktop/Bucknell/2_Scholarship/Collaboration_Ben/Trap_Door_Sims/Trap_door.txt'
# CSV_PATH = '/mnt/c/Users/ahf009/Desktop/Bucknell/2_Scholarship/Collaboration_Ben/Trap_Door_Sims/Trap_door.csv'

# TXT_PATH = '/mnt/c/Users/ahf009/Desktop/Bucknell/2_Scholarship/Collaboration_Enrique/Trap_Door_MCC.txt'
# CSV_PATH = '/mnt/c/Users/ahf009/Desktop/Bucknell/2_Scholarship/Collaboration_Enrique/Trap_Door_MCC.csv'

# TXT_PATH = '/Users/enriquedelcastillo/desktop/GEOSPH/inputfiles/' \
# 'PySPH_compressionalfault.txt'
# CSV_PATH = '/Users/enriquedelcastillo/desktop/GEOSPH/inputfiles/' \
# 'NewFault_theta60_onelayersand_AGpaper.csv'

# TXT_PATH = '/Users/blair/pysph/pysph/GEOSPH_stable_Blair2/Inputfiles/' \
#           'PySPH_simpleshear_V1_dp1_VerMCC.txt'
# CSV_PATH = '/Users/blair/pysph/pysph/GEOSPH_stable_Blair2/Inputfiles/' \
#           'PySPH_simpleshear_V2_dp1_VerMCC_tensile.csv'

# TXT_PATH = '/Users/blair/pysph/pysph/GEOSPH_stable_Blair2/Inputfiles/' \
#            'PySPH_simpleshear_V1_dp1_VerMCC.txt'
# CSV_PATH = '/Users/blair/pysph/pysph/GEOSPH_stable_Blair2/Inputfiles/' \
#            'PySPH_simpleshear_V2_dp1_VerMCC_CRM.csv'

#TXT_PATH = '/Users/enriquedelcastillo/desktop/GEOSPH_multimodel/Inputfiles/' \
#           'embankment_multi_p5.txt'
#CSV_PATH = '/Users/enriquedelcastillo/desktop/GEOSPH_multimodel/Inputfiles/' \
#           'embankment_multi_stress_point1_sizep5_part2MCC+DP_25m_NORMAL.csv'

#TXT_PATH = '/Users/enriquedelcastillo/desktop/GEOSPH_drained/Inputfiles/' \
#          'biaxial_textfile.txt'
#CSV_PATH = '/Users/enriquedelcastillo/desktop/GEOSPH_drained/Inputfiles/' \
#           'Biaxial_confining50kpa_nosides_MCC_p003_corrected_pc050kpa.csv'
#TXT_PATH = 'Inputfiles/FLUME/' \
#          'flume.txt'
#CSV_PATH = 'Inputfiles/FLUME/' \
#           'flume_angle45_phi35.csv'
TXT_PATH = 'Inputfiles/Calderas/Caldera_init2D.txt'
CSV_PATH = 'Inputfiles/Calderas/no_bottom_caldera_extendedbotbound_topo.csv'        
# =============================================================================
# =============================================================================


# Define an 'Application' class by subclassing the pysph.solver.application.
# Application class
class SDPySPHApplication(Application):

    def initialize(self):





        """ Initialize user defined parameters for the simulation, f.e.
        constants, etc. One can write a TXT file containing the input values
        for each field defined. For an explanation of the formatting rules,
        look at the docstring for io_utils.import_simulation_parameters. If a
        file path is not provided, define default values
        """

        # Import parameters from a TXT file if available
        if os.path.exists(TXT_PATH):
            sim_params = import_simulation_parameters(TXT_PATH)
            self.dp = float(sim_params['dp'])  # Initial particle distance
            self.kh = float(sim_params['kh'])  # Smoothing length factor
            self.tf = float(sim_params['simTime'])  # Final simulation time
            self.time_step = float(sim_params['stepSize'])  # Initial step
            self.sim_dim = int(sim_params['simDim'])  # Space dimensions
            self.kgc = int(sim_params['CorrNorm'])  # Gradient correction
            self.kgco = int(sim_params['NOrder'])  # Correction order
            self.nnps = sim_params['NNPS'].lower()  # Type of NNPS algorithm
            self.xsph = int(sim_params['XSPH'])  # Use XSPH correction
            self.alpha = float(sim_params['alpha'])  # Monaghan alpha
            self.c0 = float(sim_params['c'])  # Initial sound speed
            self.epsilon = float(sim_params['epsilon'])  # Monaghan eps
            self.as_epsilon = float(sim_params['as_epsilon'])  # A. Stress
            self.pbc = int(sim_params['PBC'])  # Is periodic problem?
            self.pbcx = int(sim_params['PBCX'])  # Periodic in x?
            self.pbcy = int(sim_params['PBCY'])  # Periodic in y?
            self.pbcz = int(sim_params['PBCZ'])  # Periodic in z?
            self.damp_time = float(sim_params['eqTime'])  # Solution damp

            # Monaghan artificial viscosity coefficient beta
            if 'beta' in sim_params:
                self.beta = float(sim_params['beta'])
            else:
                self.beta = 2.0*self.alpha

            # Type of kernel to use
            kernel_choice = sim_params['kernel'].lower()
            if kernel_choice == 'w':
                self.kernel = WendlandQuintic
            elif kernel_choice == 's':
                self.kernel = CubicSpline
            else:
                self.kernel = WendlandQuintic

            # Type of integrator
            integrator_choice = sim_params['integration'].lower()
            if integrator_choice == 'e':
                self.integrator = MyEulerIntegrator
            else:
                self.integrator = MyEulerIntegrator  # Not implemented yet!

            # Periodic domain dimensions
            if self.pbcx == 1 or self.pbcy == 1 or self.pbcz == 1:
                self.pbc = 1
            if self.pbc == 1:
                if 'xmin' not in sim_params or 'xmax' not in sim_params:
                    print("Must provide PB dimensions: xmin, xmax")
                    print("")
                    exit()
                else:
                    self.xmin = float(sim_params['xmin'])
                    self.xmax = float(sim_params['xmax'])
                if 'ymin' not in sim_params or 'ymax' not in sim_params:
                    print("Must provide PB dimensions: ymin, ymax")
                    print("")
                    exit()
                else:
                    self.ymin = float(sim_params['ymin'])
                    self.ymax = float(sim_params['ymax'])

                if 'zmin' not in sim_params or 'zmax' not in sim_params:
                    print("Must provide PB dimensions: zmin, zmax")
                    print("")
                    exit()
                else:
                    self.zmin = float(sim_params['zmin'])
                    self.zmax = float(sim_params['zmax'])

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # If no TXT file (or invalid path), ASSIGNS THE DEFAULTS FOR MOST
        #  SIMULATION INPUT VARIABLES.
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        else:
            self.dp = 0.1
            self.kh = 2.0
            self.tf = 1.0
            self.time_step = 0.01
            self.sim_dim = int(2)
            self.kernel = CubicSpline
            self.kgc = int(0)
            self.kgco = int(0)
            self.nnps = 'll'
            self.xsph = int(1)
            self.integrator = MyEulerIntegrator
            self.alpha = 0.2
            self.beta = 0.4
            self.c0 = 500.0
            self.epsilon = 0.5
            self.as_epsilon = 0.5
            self.pbc = int(0)
            self.pbcx = int(0)
            self.pbcy = int(0)
            self.pbcz = int(0)
            self.damp_time = float(0)

            print('TXT file not found! Using input defaults.')
            print()

        # Initial smoothing length
        self.h0 = self.kh*self.dp

        # Gravity acceleration
        self.gravity = -10.0

        # Constitutive parameters
        self.c_model = int(1)
        self.y_criterion = int(2)
        self.eta = 0.0
        self.sy0 = 1e16
        self.h_mod0 = 0.0

        # Confining stress
        self.sigma_c = 0.0

        # Output frequency
        self.out_freq = int(1)

        # Debugging option
        self.debug = int(0)

        # Boundary particle for debugging
        self.bp = int(0)

        # Initialize particles for stability
        self.init_parts = int(0)

    def add_user_options(self, group):
        """ User specified command line options, i.e. sys.argv calls. This is
        parsed before running the simulation such that the values passed by the
        user can be used to configure the application.

        Parameters
        -----------
        :param group: internal class of PySPH

        Output
        -----------
        :return: None - these options will be available when running the
        simulation from the command line
        """

        group.add_argument(
            "--dp", action="store", type=float, dest="dp", default=self.dp,
            help="Initial interparticle distance."
        )

        group.add_argument(
            "--kh", action="store", type=float, dest="kh", default=self.kh,
            help="Smoothing length factor that multiplies dp to obtain h."
        )

        group.add_argument(
            "--sigma_c", action="store", type=float, dest="sigma_c",
            default=0.0,
            help="Confining stress for triaxial-type BCs. (Default: 0.0)"
        )       

        group.add_argument(
            "--simdim", action="store", type=int, dest="dim",
            default=self.sim_dim,
            help="Spatial dimensions of the problem (2 for 2D and 3 for 3D)."
        )

        group.add_argument(
            "--kernel-grad-correct", action="store", type=int, dest="kgc",
            default=self.kgc,
            help="Select whether to correct the kernel derivative "
                 "(0 = 'No', 1 = 'Yes')."
        )

        group.add_argument(
            "--kernel-grad-correct_order", action="store", type=int,
            dest="kgco", default=self.kgco,
            help="Select the order of correction for the kernel derivative "
                 "(Orders: 0, 1 or 2)."
        )

        group.add_argument(
            "--xsph", action="store", type=int, dest="xsph",
            default=self.xsph,
            help="Select whether to use XSPH to displace particles "
                 "(0 = 'No', 1 = 'Yes')."
        )

        group.add_argument(
            "--integrator", action="store", type=str, dest="integrator",
            default='E',
            help="Select which time integration scheme to use "
                 "(E = Forward Euler)."
        )

        group.add_argument(
            "--monaghan-alpha", action="store", type=float, dest="alpha",
            default=self.alpha,
            help="Monaghan's artificial viscosity coefficient alpha."
        )

        group.add_argument(
            "--monaghan-beta", action="store", type=float, dest="beta",
            default=None,
            help="Monaghan's artificial viscosity coefficient beta."
        )

        group.add_argument(
            "--c0", action="store", type=float, dest="c0", default=self.c0,
            help="Reference numerical sound speed."
        )

        group.add_argument(
            "--xsph-epsilon", action="store", type=float, dest="epsilon",
            default=self.epsilon, help="Monaghan's XSPH coefficient."
        )

        group.add_argument(
            "--monaghan-as-epsilon", action="store", type=float,
            dest="as_epsilon", default=self.as_epsilon,
            help="Monaghan's artificial stress scaling coefficient."
        )

        group.add_argument(
            "--pbc", action="store", type=int, dest="pbc", default=self.pbc,
            help="Select whether the problem has periodic boundary conditions "
                 "(0 = 'No', 1 = 'Yes')."
        )

        group.add_argument(
            "--pbc-x", action="store", type=int, dest="pbcx",
            default=self.pbcx,
            help="Whether the problem is periodic in the x-direction "
                 "(0 = 'No', 1 = 'Yes')."
        )

        group.add_argument(
            "--pbc-y", action="store", type=int, dest="pbcy",
            default=self.pbcy,
            help="Whether the problem is periodic in the y-direction "
                 "(0 = 'No', 1 = 'Yes')."
        )

        group.add_argument(
            "--pbc-z", action="store", type=int, dest="pbcz",
            default=self.pbcz,
            help="Whether the problem is periodic in the z-direction "
                 "(0 = 'No', 1 = 'Yes')."
        )

        group.add_argument(
            "--damp-time", action="store", type=float, dest="damp_time",
            default=self.damp_time,
            help="Final time for which the solution must be damped."
        )

        group.add_argument(
            "--kw", action="store", type=float, dest="kw", default=2.0,
            help="Kernel radius coefficient used to multiply the smoothing "
                 "length: r0 = kw * h0."
        )

        group.add_argument(
            "--gravity", action="store", type=float, dest="g",
            default=self.gravity, help="Gravity acceleration modulus."
        )

        group.add_argument(
            "--eta", action="store", type=float, dest="eta", default=self.eta,
            help="Dynamic viscosity of the material."
        )

        group.add_argument(
            "--c_model", action="store", type=int, dest="c_model",
            default=self.c_model,
            help="Constitutive model for the material. "
                 "0 = Elastic, 1 = Elastoplastic, 2 = Elasto-viscoplastic."
                 "(Default: 1)"
        )

        group.add_argument(
            "--y_criterion", action="store", type=int, dest="y_criterion",
            default=self.y_criterion,
            help="Yield criterion if the constitutive model is not 'EL'. "
                 "1 = Von Mises, 2 = Drucker-Prager, 3 = MCC, 4 = Damage. (Default: 2)"
        )

        group.add_argument(
            "--debug", action="store", type=int, dest="debug",
            default=self.debug,
            help="Activate printing statements for debugging."
        )

        group.add_argument(
            "--debug-bound", action="store", type=int, dest="bp",
            default=self.bp,
            help="Boundary particle to debug (print stress tensor)."
        )

        group.add_argument(
            "--init-parts", action="store", type=int, dest="init_parts",
            default=int(0),
            help="Perform initialization to find the most stable initial "
                 "particle configuration. (Default: 0 - False)"
        )

    def consume_user_options(self):
        """ This is called after the command line arguments are parsed and can
        be accessed in self.options. This is meant to be overridden by the user
        to setup any internal variables that depend on the command line
        arguments passed.
        """

        # Assign changes to simulation parameters due to command line input
        if self.options.dp != self.dp:
            self.dp = self.options.dp

        if self.options.kh != self.kh:
            self.kh = self.options.kh

        if self.options.dim != self.sim_dim:
            self.sim_dim = self.options.dim

        if self.options.kgc != self.kgc:
            self.kgc = self.options.kgc

        if self.options.kgco != self.kgco:
            self.kgco = self.options.kgco

        if self.options.xsph != self.xsph:
            self.xsph = self.options.xsph

        integrator_choice = self.options.integrator
        if integrator_choice != self.integrator:
            if integrator_choice.lower() == 'e':
                self.integrator = MyEulerIntegrator

        if self.options.alpha != self.alpha:
            self.alpha = self.options.alpha

        if self.options.beta is not None:
            self.beta = self.options.beta

        if self.options.c0 != self.c0:
            self.c0 = self.options.c0

        if self.options.epsilon != self.epsilon:
            self.epsilon = self.options.epsilon

        if self.options.as_epsilon != self.as_epsilon:
            self.as_epsilon = self.options.as_epsilon

        if self.options.pbc != self.pbc:
            self.pbc = self.options.pbc

            # Confining pressure to be applied to all particles
        if self.options.sigma_c is not None:
            self.sigma_c = self.options.sigma_c

        if self.options.pbcx != self.pbcx:
            self.pbcx = self.options.pbcx

        if self.options.pbcy != self.pbcy:
            self.pbcy = self.options.pbcy

        if self.options.pbcz != self.pbcz:
            self.pbcz = self.options.pbcz

        if self.pbcx == 1 or self.pbcy == 1 or self.pbcz == 1:
            self.pbc = 1
        elif self.pbcx == 0 and self.pbcy == 0 and self.pbcz == 0:
            self.pbc = 0

        if self.options.damp_time != self.damp_time:
            self.damp_time = self.options.damp_time

        if self.options.g != self.gravity:
            self.gravity = self.options.g

        # Reference kernel radius coefficient and radius
        self.kw = self.options.kw
        self.r0 = self.kw * self.h0

        # Time step and final simulation time
        if self.options.time_step is not None:
            self.time_step = self.options.time_step
        if self.options.final_time is not None:
            self.tf = self.options.final_time

        # Number of damping steps
        self.n_damp = int(self.damp_time / self.time_step)
        if self.options.n_damp is not None:
            self.n_damp = self.options.n_damp

        # Kernel choice
        kernel_choice = self.options.kernel
        if kernel_choice is not None:
            kernel_choice = kernel_choice.lower()
            if kernel_choice == 'wendlandquinticc6':
                self.kernel = WendlandQuinticC6
            elif kernel_choice == 'cubicspline':
                self.kernel = CubicSpline
            else:
                self.kernel = WendlandQuintic

        # Output dumping frequency
        if self.options.freq is not None:
            self.out_freq = self.options.freq

        # Constitutive model parameters
        if self.options.c_model != self.c_model:
            self.c_model = self.options.c_model

        if self.options.y_criterion != self.y_criterion:
            self.y_criterion = self.options.y_criterion

        # Dynamic viscosity
        if self.options.eta != self.eta:
            self.eta = self.options.eta

        # Debugging option
        if self.options.debug != self.debug:
            self.debug = self.options.debug

        # Boundary particle for debugging
        if self.options.bp != self.bp:
            self.bp = self.options.bp

        # Perform particle initialization
        if self.options.init_parts != 0:
            self.init_parts = int(1)

        # Check time step
        force = 1.0
        old_dt = self.time_step
        if self.gravity != 0.0:
            force = abs(self.gravity)
        self.time_step = min(
            self.time_step,
            0.2*self.h0 / self.c0,
            0.2*sqrt(self.h0 / force),
        )
        print("\nInitial time step: ", self.time_step)

        # Correct output frequency if necessary
        self.out_freq *= int(old_dt / self.time_step)
        print("New output frequency: ", self.out_freq)

    # This is a mandatory method.
    def create_particles(self):
        """ This method generates the ParticleArrays necessary for the PySPH
        framework to run. It reads a CSV file containing all the information
        properties and geometric properties of each particle and returns a list
        of PySPH ParticleArrays.

        :return: returns a list of PySPH ParticleArrays
        """

        # Read particles properties and simulation parameters from files
        part_arr = import_parts_data(CSV_PATH)

        # Header with particle fields in the CSV file
        header = get_csv_header(CSV_PATH)

        # Create the dictionary of all particles properties
        part_data = separate_particle_data_arrays(part_arr, header)

        # Select particles by type and model ('tag'/labels in csv file)
        # and associate their names with those indices
        types_arr = part_data['type']
        sediment_idx = np.where(types_arr == 1)[0]
        boundary_idx = np.where(types_arr == 0)[0]
        barrier_idx = np.where(types_arr == 2)[0]
        part_types = {'sediment': sediment_idx, 'boundary': boundary_idx,
                      'barrier': barrier_idx}

        # Initialize the ParticleArrays and add them to a list that will be
        #  returned
        sediment = get_particle_array(name='sediment')
        boundary = get_particle_array(name='boundary')
        barrier = get_particle_array(name='barrier')
        pas = [sediment, boundary, barrier]

        # Populate each PySPH ParticleArray in pas with new properties and
        #  corresponding values, except stress and strain.
        for pa_type in pas:

            for key in part_data.keys():
                if (key[0] == 's' or key[0] == 'e') and len(key) == 3:
                    continue
                pa_type.add_property(
                    key,
                    default=0.0,
                    data=part_data[key][part_types[pa_type.name]]
                )

            # Miscellaneous operations
            pa_type.align_particles()
            num_parts = pa_type.get_number_of_particles()

            # Assign smoothing length to all particles
            h_arr = self.h0 * np.ones(num_parts)
            pa_type.h = h_arr

            # Add kernel sum property
            pa_type.add_property('wsum', default=1.0)

            # Add density rate
            pa_type.add_property('arho', default=0.0)

            # Add Cauchy stress tensor
            sigma = np.zeros(9*num_parts)
            sig_keys = ['sxx', 'sxy', 'sxz', 'sxy', 'syy', 'syz', 'sxz', 'syz',
                        'szz']

            for i, key in enumerate(sig_keys):
                if key in part_data.keys():
                    sigma[i::9] = part_data[key][part_types[pa_type.name]]
                else:
                    print("Key \'%s\' not found!" % key)
                    exit()

            pa_type.add_property('sigma', default=0.0, data=sigma, stride=9)

            # Bulk and shear elastic moduli
            young = part_data['young'][part_types[pa_type.name]]
            poisson = part_data['poisson'][part_types[pa_type.name]]
            bulk = young / (3.0 * (1.0 - 2.0 * poisson))
            shear = young / (2 * (1 + poisson))
            pa_type.add_property('bulk', default=1.0, data=bulk)
            pa_type.add_property('shear', default=1.0, data=shear)

            # Sound speed
            rho0 = part_data['rho'][part_types[pa_type.name]]
            cs0 = np.maximum(np.sqrt(3*bulk / rho0), np.sqrt(young / rho0))
            pa_type.add_property('cs', default=self.c0, data=cs0)

            # Artificial Stress
            pa_type.add_property('asig', default=0.0, stride=9)

            # Add body forces
            body = np.zeros(3*num_parts)
            bf_keys = ['fx', 'fy', 'fz']

            for i, key in enumerate(bf_keys):
                if key in part_data.keys():
                    body[i::3] = part_data[key][part_types[pa_type.name]]
                else:
                    print("Key \'%s\' not found!" % key)
                    exit()

            pa_type.add_property('f', default=0.0, data=body, stride=3)

            # =================================================================
            # Add properties to sediment particles only
            if pa_type.name == 'sediment':

                # Deviatoric stress tensor
                pa_type.add_property('sigma_dev', default=0.0, stride=9)

                # Trial stress tensor
                pa_type.add_property('sigma_tr', default=0.0, data=sigma,
                                     stride=9)

                # Stress rate
                pa_type.add_property('sigma_dot', default=0.0, stride=9)

                # Deviatoric stress invariant, q
                pa_type.add_property('q', default=0.0)

                # Artificial stress acceleration, as_a
                pa_type.add_property('as_a', default=0.0, stride=3)

                # Kernel value for dp
                pa_type.add_property('wdp', default=0.0)

                # Viscosity parameter
                pa_type.add_property('eta', default=self.eta)

                # Accumulated displacement
                pa_type.add_property('disp', default=0.0, stride=3)

                # Add elastic strain tensors
                eps_e = np.zeros(9*num_parts)
                e_keys = ['exx', 'exy', 'exz', 'exy', 'eyy', 'eyz', 'exz',
                          'eyz', 'ezz']
                for i, key in enumerate(e_keys):
                    if key in part_data.keys():
                        eps_e[i::9] = part_data[key][part_types[pa_type.name]]
                    else:
                        print("Key \'%s\' not found!" % key)
                        exit()

                # Initialize plastic and total strain tensors
                eps_p = np.zeros(9*num_parts)
                eps = np.copy(eps_e)

                # Make sure that Eps_zz = 0 if plane-strain
                if self.sim_dim == 2:
                    eps_p[8::9] = -eps_e[8::9]
                    eps[8::9] = 0.0

                # n_i
                # n = np.zeros(9*num_parts)
                # n_keys = ['nxx', 'nxy', 'nxz', 'nxy', 'nyy', 'nyz', 'nxz', 'nyz',
                #         'nzz']

                # for i, key in enumerate(n_keys):
                #     if key in part_data.keys():
                #         n[i::9] = part_data[key][part_types[pa_type.name]]
                #     else:
                #         print("Key \'%s\' not found!" % key)
                #         exit()

                # pa_type.add_property('n', default=0.0, data=n, stride=9)

                # Deformation vectors
                pa_type.add_property('eps', default=0.0, data=eps, stride=9)
                pa_type.add_property('eps_e', default=0.0, data=eps_e,
                                     stride=9)
                pa_type.add_property('eps_p', default=0.0, data=eps_p,
                                     stride=9)
                pa_type.add_property('eps_dot', default=0.0, stride=9)
                pa_type.add_property('eps_p_dot', default=0.0, stride=9)
                pa_type.add_property('spin_dot', default=0.0, stride=9)
                pa_type.add_property('ep_acc', default=0.0)
                # pa_type.add_property('ep_acc_v', default=0.0)
                # pa_type.add_property('ep_acc_d', default=0.0)
                ep_acc_v = part_data['ep_acc_v'][part_types[pa_type.name]]
                ep_acc_d = part_data['ep_acc_d'][part_types[pa_type.name]]
                pa_type.add_property('ep_acc_v', default=0.0, data=ep_acc_v)
                pa_type.add_property('ep_acc_d', default=0.0, data=ep_acc_d)

                # Add other properties to sediment particle arrays
                properties = ['ke', 'pe', 'se', 'te']
                for prop in properties:
                    pa_type.add_property(prop, default=0.0)

                # Flag property
                pa_type.add_property('flag', default=0, type='int')

                # Kernel derivative correction matrix
                m_mat = np.zeros(9 * num_parts)
                m_mat[::9] = 1.0
                m_mat[4::9] = 1.0
                m_mat[8::9] = 1.0
                pa_type.add_property('m_mat', default=0.0, data=m_mat,
                                     stride=9)

                # Add Seismic accelerogram data from San Fernando 1971 earthquake
                address = ("SF_horizontal_intp")
                a_h = np.loadtxt(address, delimiter=',')
                address2 = ("SF_vertical_intp")
                a_v = np.loadtxt(address2, delimiter=',')
                seis_time = a_h[:, 0]
                a_hor = a_h[:, 1]
                a_ver = a_v[:, 1]

                pa_type.add_constant('a_ver', data=a_ver)
                pa_type.add_constant('a_hor', data=a_hor)
                pa_type.add_constant('seis_time', data=seis_time)
               # nstep = IntArray()
                # nstep=[0]
                pa_type.add_property('nstep', 'int', data=0)
                print(pa_type.get_carray('nstep').get_c_type())

                # Add caldera pressure time history
                address = ("pChange_intp")
                p_time = np.loadtxt(address, delimiter=',')
                p_hist = p_time[:, 0]
                delta_p = p_time[:, 1]
                pa_type.add_constant('p_hist', data=p_hist)
                pa_type.add_constant('delta_p', data=delta_p)


                # print(pa_type.nstep)

                # Drucker-Prager parameters. sy is the uniaxial yield stress.
                #  This ensures consistency between VM and DP models for
                #  phi = 0.
                if self.y_criterion == 2:
                    phi = part_data['phi'][part_types[pa_type.name]
                                           ] * pi / 180.0
                    cohesion = part_data['cohesion'][part_types[pa_type.name]]
                    psi = part_data['psi'][part_types[pa_type.name]
                                           ] * pi / 180.0
                    h_mod = part_data['h_mod'][part_types[pa_type.name]]
                    ac = sqrt(2.0/3.0)

                    if self.sim_dim == 2:
                        sy = sqrt(3)*cohesion
                        aphi = sqrt(6)*np.tan(phi) / \
                            np.sqrt(3 + 4*np.power(np.tan(phi), 2))
                        apsi = sqrt(6)*np.tan(psi) / \
                            np.sqrt(3 + 4*np.power(np.tan(psi), 2))
                        if self.y_criterion == 2:
                            ac = sqrt(2)/np.sqrt(3 + 4 *
                                                 np.power(np.tan(phi), 2))
                    else:
                        sy = 2*cohesion

                        # Fit to the MCYC through the outer edges ("-" compression)
                        aphi = 2*sqrt(6)*np.sin(phi) / (3 - np.sin(phi))
                        apsi = 2*sqrt(6)*np.sin(psi) / (3 - np.sin(psi))
                        if self.y_criterion == 2:
                            ac = sqrt(6)*np.cos(phi) / (3 - np.sin(phi))

                    pa_type.add_property('aphi', default=0.0, data=aphi)
                    pa_type.add_property('apsi', default=0.0, data=apsi)
                    pa_type.add_property('ac', default=0.0, data=ac)
                    pa_type.add_property('sy', default=0.0, data=sy)
                    pa_type.add_property('h_mod', default=0.0, data=h_mod)

                # Modified Cam-Clay model
                if self.y_criterion == 3:
                    pc = part_data['pc'][part_types[pa_type.name]]
                    ms = part_data['ms'][part_types[pa_type.name]]
                    pt = part_data['pt'][part_types[pa_type.name]]
                    void_ratio = part_data['void_ratio'][part_types[pa_type.name]]
                    void_ref = part_data['void_ref'][part_types[pa_type.name]]
                    lambda_mcc = part_data['lambda_mcc'][part_types[pa_type.name]]
                    kappa_mcc = part_data['kappa_mcc'][part_types[pa_type.name]]

                    pa_type.add_property('pc', default=0.0, data=pc)
                    pa_type.add_property('pt', default=0.0, data=pt)
                    pa_type.add_property('ms', default=0.0, data=ms)
                    pa_type.add_property(
                        'void_ratio', default=0.0, data=void_ratio)
                    pa_type.add_property(
                        'void_ref', default=0.0, data=void_ref)
                    pa_type.add_property(
                        'lambda_mcc', default=0.0, data=lambda_mcc)
                    pa_type.add_property(
                        'kappa_mcc', default=0.0, data=kappa_mcc)

                # Continuum damage model
                if self.y_criterion == 4:
                    kd = part_data['kd'][part_types[pa_type.name]]
                    kappa_o = part_data['kappa_o'][part_types[pa_type.name]]
                    betad = part_data['betad'][part_types[pa_type.name]]
                    alphad = part_data['alphad'][part_types[pa_type.name]]
                    dam = part_data['dam'][part_types[pa_type.name]]

                    pa_type.add_property('kd', default=0.0, data=kd)
                    pa_type.add_property('kappa_o', default=0.0, data=kappa_o)
                    pa_type.add_property('betad', default=0.0, data=betad)
                    pa_type.add_property('alphad', default=0.0, data=alphad)
                    pa_type.add_property('dam', default=0.0, data=dam)

                # Continuum chemical-damage model
                if self.y_criterion == 5:
                    kd = part_data['kd'][part_types[pa_type.name]]
                    kappa_o = part_data['kappa_o'][part_types[pa_type.name]]
                    betad = part_data['betad'][part_types[pa_type.name]]
                    alphad = part_data['alphad'][part_types[pa_type.name]]
                    dam = part_data['dam'][part_types[pa_type.name]]
                    dam_m = part_data['dam_m'][part_types[pa_type.name]]
                    dam_c = part_data['dam_c'][part_types[pa_type.name]]
                    ad = part_data['ad'][part_types[pa_type.name]]
                    m_inf = part_data['m_inf'][part_types[pa_type.name]]
                    eps_o = part_data['eps_o'][part_types[pa_type.name]]
                    etad = part_data['etad'][part_types[pa_type.name]]
                    eps_eq = part_data['eps_eq'][part_types[pa_type.name]]

                    pa_type.add_property('kd', default=0.0, data=kd)
                    pa_type.add_property('kappa_o', default=0.0, data=kappa_o)
                    pa_type.add_property('betad', default=0.0, data=betad)
                    pa_type.add_property('alphad', default=0.0, data=alphad)
                    pa_type.add_property('dam', default=0.0, data=dam)
                    pa_type.add_property('dam_m', default=0.0, data=dam_m)
                    pa_type.add_property('dam_c', default=0.0, data=dam_c)
                    pa_type.add_property('ad', default=0.0, data=ad)
                    pa_type.add_property('m_inf', default=0.0, data=m_inf)
                    pa_type.add_property('eps_o', default=0.0, data=eps_o)
                    pa_type.add_property('etad', default=0.0, data=etad)
                    pa_type.add_property('eps_eq', default=0.0, data=eps_eq)

                # Friction softening D-P model
                if self.y_criterion == 6:
                    phi_p = part_data['phi_p'][part_types[pa_type.name]
                                               ] * pi / 180.0
                    phi_res = part_data['phi_res'][part_types[pa_type.name]] * pi / 180.0
                    cohesion = part_data['cohesion'][part_types[pa_type.name]]
                    psi_p = part_data['psi_p'][part_types[pa_type.name]
                                               ] * pi / 180.0
                    psi_res = part_data['psi_res'][part_types[pa_type.name]] * pi / 180.0
                    h_mod = part_data['h_mod'][part_types[pa_type.name]]
                    eps_p_f = part_data['eps_p_f'][part_types[pa_type.name]]
                    sy = sqrt(3)*cohesion
                    eps_p_oct = 0
                    phi = 0
                    psi = 0

                    pa_type.add_property('sy', default=0.0, data=sy)
                    pa_type.add_property('phi_p', default=0.0, data=phi_p)
                    pa_type.add_property('phi_res', default=0.0, data=phi_res)
                    pa_type.add_property('psi_p', default=0.0, data=psi_p)
                    pa_type.add_property('psi_res', default=0.0, data=psi_res)
                    pa_type.add_property('h_mod', default=0.0, data=h_mod)
                    pa_type.add_property(
                        'eps_p_oct', default=0.0, data=eps_p_oct)
                    pa_type.add_property('eps_p_f', default=0.0, data=eps_p_f)
                    pa_type.add_property('phi', default=0.0, data=phi)
                    pa_type.add_property('psi', default=0.0, data=psi)

                # Multimodel
                if self.y_criterion == 23:
                    # Friction softening D-P model
                    phi_p = part_data['phi_p'][part_types[pa_type.name]
                                               ] * pi / 180.0
                    phi_res = part_data['phi_res'][part_types[pa_type.name]] * pi / 180.0
                    cohesion = part_data['cohesion'][part_types[pa_type.name]]
                    psi_p = part_data['psi_p'][part_types[pa_type.name]
                                               ] * pi / 180.0
                    psi_res = part_data['psi_res'][part_types[pa_type.name]] * pi / 180.0
                    h_mod = part_data['h_mod'][part_types[pa_type.name]]
                    eps_p_f = part_data['eps_p_f'][part_types[pa_type.name]]
                    sy = sqrt(3)*cohesion
                    eps_p_oct = 0
                    phi = 0
                    psi = 0
                    model = part_data['model'][part_types[pa_type.name]]

                    pa_type.add_property('model', default=0, data=model)
                    pa_type.add_property('sy', default=0.0, data=sy)
                    pa_type.add_property('phi_p', default=0.0, data=phi_p)
                    pa_type.add_property('phi_res', default=0.0, data=phi_res)
                    pa_type.add_property('psi_p', default=0.0, data=psi_p)
                    pa_type.add_property('psi_res', default=0.0, data=psi_res)
                    pa_type.add_property('h_mod', default=0.0, data=h_mod)
                    pa_type.add_property(
                        'eps_p_oct', default=0.0, data=eps_p_oct)
                    pa_type.add_property('eps_p_f', default=0.0, data=eps_p_f)
                    pa_type.add_property('phi', default=0.0, data=phi)
                    pa_type.add_property('psi', default=0.0, data=psi)

                    # Modified Cam-Clay
                    pc = part_data['pc'][part_types[pa_type.name]]
                    ms = part_data['ms'][part_types[pa_type.name]]
                    pt = part_data['pt'][part_types[pa_type.name]]
                    void_ratio = part_data['void_ratio'][part_types[pa_type.name]]
                    void_ref = part_data['void_ref'][part_types[pa_type.name]]
                    lambda_mcc = part_data['lambda_mcc'][part_types[pa_type.name]]
                    kappa_mcc = part_data['kappa_mcc'][part_types[pa_type.name]]

                    pa_type.add_property('pc', default=0.0, data=pc)
                    pa_type.add_property('pt', default=0.0, data=pt)
                    pa_type.add_property('ms', default=0.0, data=ms)
                    pa_type.add_property(
                        'void_ratio', default=0.0, data=void_ratio)
                    pa_type.add_property(
                        'void_ref', default=0.0, data=void_ref)
                    pa_type.add_property(
                        'lambda_mcc', default=0.0, data=lambda_mcc)
                    pa_type.add_property(
                        'kappa_mcc', default=0.0, data=kappa_mcc)
                    
                    # eps_p_norm time integration
                    pa_type.add_property('eps_p_norm', default=0.0)

                    
                    pa_type.add_property('fi', default=0.0, data=0.0)


                # Set output arrays for sediment PAs
                # for DP:
                pa_type.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'rho',
                                           'eps', 'eps_e', 'eps_p', 'ep_acc', 'ep_acc_v',
                                           'ep_acc_d', 'ke', 'pe', 'te', 'p', 'q',
                                           'sigma', 'gid', 'type', 'disp', 'eps_dot', 'fi',
                                           'r', 'g', 'b', 'nstep', 'void_ratio', 'pc', 'bulk','eps_p_norm','model','eps_p_oct','phi','psi','color'])
              # for damage:#
                # pa_type.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'rho',
                #                          'eps', 'eps_e', 'eps_p', 'ep_acc','ep_acc_v',
                #                           'ep_acc_d','ke', 'pe', 'te', 'p', 'q',
                #                           'sigma','gid', 'dam','dam_c','dam_m','type', 'disp', 'eps_dot',
                #                           'r', 'g', 'b','eps_eq'])
               # for DP+friction softening:
                # pa_type.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w', 'rho',
                # 'eps', 'eps_e', 'eps_p', 'ep_acc','ep_acc_v',
                #                          'ep_acc_d','ke', 'pe', 'te', 'p', 'q',
                #                          'sigma','gid','type', 'disp', 'eps_dot',
                #                          'r', 'g', 'b','eps_p_oct','phi','psi','color'])

            else:
                pa_type.set_output_arrays(['x', 'y', 'z', 'rho', 'sigma',
                                           'gid', 'type'])

            # =================================================================

            if pa_type.name == 'boundary' or pa_type.name == 'barrier':
                pa_type.add_property('du', default=0.0)

            # load balancing properties
            pa_type.set_lb_props(list(pa_type.properties.keys()))

        return pas

    # Same as 'create_equations'

    def create_solver(self):
        kernel = self.kernel(self.sim_dim)

        sediment_stepper = SedimentEulerStep(
            gravity=self.gravity, damp_time=self.damp_time,
            sim_dim=self.sim_dim, debug=self.debug
        )
        boundary_stepper = BoundaryEulerStep(self.debug, self.bp)

        integrator = MyEulerIntegrator(
            sediment=sediment_stepper, boundary=boundary_stepper,
            barrier=boundary_stepper
        )

        solver = Solver(
            kernel=kernel, dim=self.sim_dim, integrator=integrator,
            dt=self.time_step, tf=self.tf, fixed_h=True, pfreq=self.out_freq
        )
        return solver

    # This method must be overloaded if not using a Scheme instance.
    def create_equations(self):
        if self.init_parts == 0:
            equations = [

                # Group 1
                Group(
                    equations=[
                        KernelSum(
                            dest='boundary',
                            sources=['sediment'],
                            debug=self.debug,
                            bp=self.bp,
                        ),
                        KernelSum(
                            dest='barrier',
                            sources=['sediment'],
                            debug=self.debug,
                            bp=self.bp,
                        ),
                        KernelSum(
                            dest='sediment',
                            sources=['sediment', 'boundary', 'barrier'],
                            debug=self.debug,
                            bp=self.bp,
                        ),
                        KernelGradLMatrix(
                            dest='sediment',
                            sources=['sediment', 'boundary', 'barrier'],
                            kgc=self.kgc,
                            sim_dim=self.sim_dim,
                            debug=self.debug,
                        )
                    ],
                    real=True
                ),

                # Group 2
                Group(
                    equations=[
                        KernelGradCorrect(
                            dest='sediment',
                            sources=['sediment', 'boundary', 'barrier'],
                            kgc=self.kgc, sim_dim=self.sim_dim,
                        ),

                        BoundaryStress(
                            dest='boundary',
                            sources=['sediment'],
                            sim_dim=self.sim_dim,
                            debug_bound=self.bp,
                        ),
                        BoundaryStress(
                            dest='barrier',
                            sources=['sediment'],
                            sim_dim=self.sim_dim,
                            debug_bound=self.bp,
                        ),
                         ConfiningBCtreatment(
                            dest='sediment',
                            sources=['sediment', 'boundary', 'barrier'],
                            debug_bound=self.bp,
                        ),
                    ],
                    real=True
                ),

                # Group 3
                Group(
                    equations=[
                        DeformationRates(
                            dest='sediment',
                            sources=['sediment', 'boundary', 'barrier'],
                            debug=self.debug,
                        ),
                        ArtStress(
                            dest='sediment',
                            sources=['sediment', 'boundary', 'barrier'],
                            as_eps=self.as_epsilon,
                            dp=self.dp,
                            c_model=self.c_model,
                            debug=self.debug,
                        ),
                    ],
                    real=True
                ),

                # Group 4
                Group(
                    equations=[
                        DensityEquation(
                            dest='sediment',
                            sources=['sediment', 'boundary', 'barrier'],
                            debug=self.debug,
                        ),
                        TrialStress(
                            dest='sediment',
                            sources=['sediment'],
                            debug=self.debug,
                        ),
                        TrialStressDecomposition(
                            dest='sediment',
                            sources=['sediment'],
                            debug=self.debug,
                        ),
                        # DPSolver(
                        #    dest='sediment',
                        #    sources=None,
                        #    c_model=self.c_model,
                        #    y_criterion=self.y_criterion,
                        #    debug=self.debug,
                        # ),
                        # DPSolverSoft(
                        #     dest='sediment',
                        #     sources=None,
                        #     c_model=self.c_model,
                        #     y_criterion=self.y_criterion,
                        #     debug=self.debug,
                        # ),
                        # MCCSolver_Approx_CPP(
                        #    dest='sediment',
                        #    sources=None,
                        #    nu = 0.3,
                        #    c_model=self.c_model,
                        #    y_criterion=self.y_criterion,
                        #    tol=1e-5,
                        #    max_iter=100,
                        #    debug=self.debug,
                        # ),
                        MultiModel(
                            dest='sediment',
                            sources=None,
                            nu=0.3,
                            c_model=self.c_model,
                            tol=1e-5,
                            max_iter=100,
                            debug=self.debug,
                        ),
                        #MultiModelExact(
                        #    dest='sediment',
                        #    sources=None,
                        #    nu=0.3,
                        #    c_model=self.c_model,
                        #    tol=1e-5,
                        #    max_iter=100,
                        #    debug=self.debug,
                        #),
                        # MCCSolver_Exact_CPP(
                        #    dest='sediment',
                        #    sources=None,
                        #    nu = 0.3,
                        #    c_model=self.c_model,
                        #    y_criterion=self.y_criterion,
                        #    tol=1e-3,
                        #    max_iter=500,
                        #    debug=self.debug,
                        # ),
                        # MCCSolverCRM(
                        #    dest='sediment',
                        #    sources=None,
                        #     nu = 0.3,
                        #     c_model=self.c_model,
                        #     y_criterion=self.y_criterion,
                        #     tol=1e-3,
                        #     max_iter=1000,
                        #     debug=self.debug,
                        # ),
                        # MCCSolver_Exact_Tensile_CPP(
                        #    dest='sediment',
                        #    sources=None,
                        #    nu = 0.3,
                        #    c_model=self.c_model,
                        #    y_criterion=self.y_criterion,
                        #    tol=1e-5,
                        #    max_iter=100,
                        #    debug=self.debug,
                        # ),
                        # DModel(
                        #   dest='sediment',
                        #   sources=None,
                        #    nu = 0.3,
                        #    c_model=self.c_model,
                        #    y_criterion=self.y_criterion,
                        #    debug=self.debug,
                        # ),
                        # DModelC(
                        #   dest='sediment',
                        #   sources=None,
                        #    nu = 0.3,
                        #    c_model=self.c_model,
                        #    y_criterion=self.y_criterion,
                        #    debug=self.debug,
                        # ),
                        #StressRegularization(
                        #    dest='sediment',
                        #    sources=['sediment'],
                        #    debug=self.debug,
                        #),
                        MomentumEquation(
                            dest='sediment',
                            sources=['sediment', 'boundary', 'barrier'],
                            sim_dim=self.sim_dim,
                            sigma_c=self.sigma_c,
                            debug=self.debug,
                        ),
                        ArtVisc(
                            dest='sediment',
                            sources=['sediment', 'boundary', 'barrier'],
                            alpha=self.alpha,
                            beta=self.beta,
                            debug=self.debug,
                        ),
                        XSph(
                            dest='sediment',
                            sources=['sediment', 'boundary', 'barrier'],
                            eps=self.epsilon,
                            freq=1,
                            debug=self.debug,
                        ),
                    ],
                    real=True
                ),
            ]

        # For a different scheme (e.g., multi-phase flow or initialization)
        else:
            equations = []
        return equations

    def create_nnps(self):
        if self.pbc == 1:
            domain = DomainManager(
                xmin=self.xmin, xmax=self.xmax,
                ymin=self.ymin, ymax=self.ymax,
                zmin=self.zmin, zmax=self.zmax,
                periodic_in_x=self.pbcx,
                periodic_in_y=self.pbcy,
                periodic_in_z=self.pbcz
            )
        else:
            domain = None

        if self.nnps.upper() == 'LL':  # Linked-list NNPS
            nps = LinkedListNNPS(
                dim=self.sim_dim,
                particles=self.particles,
                radius_scale=self.kw,
                cache=False,
                domain=domain
            )
        else:
            nps = LinkedListNNPS(
                dim=self.sim_dim,
                particles=self.particles,
                radius_scale=self.kw,
                cache=False,
                domain=domain
            )
        return nps

    # Post-process output and other results management methods
    def post_process(self, info_filename):
        in_path = os.path.abspath(self.options.output_dir)
        dir_path = os.path.join(in_path, 'VTU')

        # Create a vtu directory if not there yet.
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        # Save output files to the VTU directory with VTU format.
        convert_pysph_output(in_path, dir_path, ftype=1, version='S')


# After setting up the framework, instantiate the class (which includes the
#  solver) and run it to get the solution.
if __name__ == '__main__':
    app = SDPySPHApplication()
    app.run()

    # Post-processing is added here
    app.post_process(app.info_filename)
