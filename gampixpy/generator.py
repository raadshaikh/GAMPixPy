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
                                self.z_init)],
                              dtype = meta_dtype)
                
        return meta_array 
