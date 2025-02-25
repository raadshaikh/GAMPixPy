import numpy as np
from gampixpy import tracks
from gampixpy.input_parsing import meta_dtype

class Generator:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        while True:
            yield self.get_sample(), self.get_meta()
    
class PointSource (Generator):
    def generate_sample_params(self):
        self.n_samples_per_point = 100000

        self.x_init = self.kwargs['x_range'][0] + (self.kwargs['x_range'][1] - self.kwargs['x_range'][0])*np.random.random()
        self.y_init = self.kwargs['y_range'][0] + (self.kwargs['y_range'][1] - self.kwargs['y_range'][0])*np.random.random()
        self.z_init = self.kwargs['z_range'][0] + (self.kwargs['z_range'][1] - self.kwargs['z_range'][0])*np.random.random()
        self.q_init = self.kwargs['q_range'][0] + (self.kwargs['q_range'][1] - self.kwargs['q_range'][0])*np.random.random()

    def get_sample(self):
        self.generate_sample_params()

        charge_points = np.stack(self.n_samples_per_point*[[self.x_init,
                                                            self.y_init,
                                                            self.z_init]])
        charge_values = np.array(self.n_samples_per_point*[self.q_init/self.n_samples_per_point,
                                                           ])                                 
        
        return tracks.Track(charge_points, charge_values)

    def get_meta(self):
        meta_array = np.array([(0, 0,
                                self.q_init,
                                self.x_init,
                                self.y_init,
                                self.z_init,
                                0,
                                0,
                                0)],
                              dtype = meta_dtype)
                
        return meta_array 

class LineSource (Generator):
    def generate_sample_params(self):
        self.n_samples_per_segment = 100000

        self.x_init = self.kwargs['x_range'][0] + (self.kwargs['x_range'][1] - self.kwargs['x_range'][0])*np.random.random()
        self.y_init = self.kwargs['y_range'][0] + (self.kwargs['y_range'][1] - self.kwargs['y_range'][0])*np.random.random()
        self.z_init = self.kwargs['z_range'][0] + (self.kwargs['z_range'][1] - self.kwargs['z_range'][0])*np.random.random()
        self.q = self.kwargs['q_range'][0] + (self.kwargs['q_range'][1] - self.kwargs['q_range'][0])*np.random.random()

        # maybe direction parameters can be passed as kwargs?
        # for now, let's just always use a uniform distribution
        # over the unit sphere
        self.theta = 2*np.pi*np.random.random()
        self.phi = np.arccos(1 - 2*np.random.random())

        self.length = self.kwargs['length_range'][0] + (self.kwargs['length_range'][1] - self.kwargs['length_range'][0])*np.random.random()
        
    def get_sample(self):
        self.generate_sample_params()

        pos_vec = np.array([self.x_init,
                            self.y_init,
                            self.z_init])
        dir_vec = np.array([np.cos(self.theta)*np.sin(self.phi),
                            np.sin(self.theta)*np.sin(self.phi),
                            np.cos(self.phi),
                            ])
        
        # charge_points = np.stack(self.n_samples_per_point*[[self.x_init,
        #                                                     self.y_init,
        #                                                     self.z_init]])
        charge_points = pos_vec + np.linspace(0, 1, self.n_samples_per_segment)[:,None]*dir_vec*self.length
        # uniform charge profile (subject to change?)
        charge_values = np.array(self.n_samples_per_segment*[self.q/self.n_samples_per_segment,
                                                           ])                                 
        
        return tracks.Track(charge_points, charge_values)

    def get_meta(self):
        meta_array = np.array([(0, 0,
                                self.q,
                                self.x_init,
                                self.y_init,
                                self.z_init,
                                self.theta,
                                self.phi,
                                self.length)],
                              dtype = meta_dtype)
                
        return meta_array 
