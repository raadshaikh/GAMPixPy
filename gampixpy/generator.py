import numpy as np
import torch

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
        self.t_init = self.kwargs['t_range'][0] + (self.kwargs['t_range'][1] - self.kwargs['t_range'][0])*np.random.random()
        self.q_init = self.kwargs['q_range'][0] + (self.kwargs['q_range'][1] - self.kwargs['q_range'][0])*np.random.random()

    def get_sample(self):
        self.generate_sample_params()

        charge_4vec = torch.tensor(self.n_samples_per_point*[[self.x_init,
                                                              self.y_init,
                                                              self.z_init,
                                                              self.t_init,
                                                              ]])
        charge_values = torch.tensor(self.n_samples_per_point*[self.q_init/self.n_samples_per_point,
                                                               ])                                 
        
        return tracks.Track(charge_4vec, charge_values)

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
        self.n_samples = 10000

        self.x_init = self.kwargs['x_range'][0] + (self.kwargs['x_range'][1] - self.kwargs['x_range'][0])*np.random.random()
        self.y_init = self.kwargs['y_range'][0] + (self.kwargs['y_range'][1] - self.kwargs['y_range'][0])*np.random.random()
        self.z_init = self.kwargs['z_range'][0] + (self.kwargs['z_range'][1] - self.kwargs['z_range'][0])*np.random.random()
        self.t_init = self.kwargs['t_range'][0] + (self.kwargs['t_range'][1] - self.kwargs['t_range'][0])*np.random.random()
        self.q = self.kwargs['q_range'][0] + (self.kwargs['q_range'][1] - self.kwargs['q_range'][0])*np.random.random()

        # maybe direction parameters can be passed as kwargs?
        # for now, let's just always use a uniform distribution
        # over the unit sphere
        self.theta = 2*torch.pi*torch.rand(1)
        self.phi = torch.arccos(1 - 2*torch.rand(1))

        self.length = self.kwargs['length_range'][0] + (self.kwargs['length_range'][1] - self.kwargs['length_range'][0])*torch.rand(1)

    def do_point_sampling(self, start_4vec, end_4vec,
                          dx, charge_per_sample):
        # point sampling with a fixed number of samples per length
        # it may be faster to do sampling another way (test in future!)
        #  - sample with fixed amount of charge
        #  - sample with fixed number of samples per segment

        segment_interval = end_4vec - start_4vec

        sample_parametric_distance = torch.linspace(0, 1, self.n_samples)
        sample_4vec = start_4vec + segment_interval*sample_parametric_distance[:,None]
        
        sample_charges = charge_per_sample*torch.ones(self.n_samples)

        return sample_4vec, sample_charges
        
    def get_sample(self):
        self.generate_sample_params()

        start_4vec = torch.tensor((self.x_init,
                                   self.y_init,
                                   self.z_init,
                                   self.t_init,
                                   ))
        dir_4vec = torch.tensor([torch.cos(torch.tensor(self.theta))*torch.sin(torch.tensor(self.phi)),
                                 torch.sin(torch.tensor(self.theta))*torch.sin(torch.tensor(self.phi)),
                                 torch.cos(torch.tensor(self.phi)),
                                 0,
                                 ])
        end_4vec = start_4vec + dir_4vec*self.length

        displacement = start_4vec[:3] - end_4vec[:3]
        dx = torch.sum(displacement**2)
        dQ = self.q/self.n_samples
        charge_4vec, charge_values = self.do_point_sampling(start_4vec,
                                                            end_4vec,
                                                            dx, dQ,
                                                            )

        return tracks.Track(charge_4vec, charge_values)

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
