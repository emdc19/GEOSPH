from pysph.sph.integrator import Integrator


class MyEulerIntegrator(Integrator):

    def one_time_step(self, t, dt):
        self.compute_accelerations()
        self.stage1()
        self.update_domain()
        self.do_post_stage(dt, 1)
