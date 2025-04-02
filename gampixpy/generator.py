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

        charge_4vec = torch.stack(self.n_samples_per_point*[[self.x_init,
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

    def do_point_sampling(self, start_4vec, end_4vec,
                          dx, charge_per_segment,
                          sample_density = 1.e-1,
                          sample_normalization = 'charge'):
        # point sampling with a fixed number of samples per length
        # it may be faster to do sampling another way (test in future!)
        #  - sample with fixed amount of charge
        #  - sample with fixed number of samples per segment

        segment_interval = end_4vec - start_4vec

        if sample_normalization == 'charge':
            # here, sample_density is [samples/unit charge]
            samples_per_segment = (charge_per_segment*sample_density).int()
        elif sample_normalization == 'length':
            # here, sample_density is [samples/unit length]
            samples_per_segment = (dx*sample_density).int()

        sample_start = torch.repeat_interleave(start_4vec, samples_per_segment, dim = 0)
        sample_interval = torch.repeat_interleave(segment_interval, samples_per_segment, dim = 0)
        sample_parametric_distance = torch.cat(tuple(torch.linspace(0, 1, samples_per_segment[i])
                                                     for i in range(samples_per_segment.shape[0])))
        sample_4vec = sample_start + sample_interval*sample_parametric_distance[:,None]
        
        sample_charges = torch.repeat_interleave(charge_per_segment/samples_per_segment, samples_per_segment)

        return sample_4vec, sample_charges
        
    def get_sample(self):
        self.generate_sample_params()

        start_4vec = torch.tensor((self.x_init,
                                   self.y_init,
                                   self.z_init,
                                   0,
                                   ))
        dir_4vec = np.array([np.cos(self.theta)*np.sin(self.phi),
                             np.sin(self.theta)*np.sin(self.phi),
                             np.cos(self.phi),
                             0,
                             ])
        end_4vec = start_4vec + dir_4vec*self.length

        displacement = start_4vec[:,:3] - end_4vec[:,:3]
        dx = torch.sum(displacement**2, dim = 1)
        dQ = self.q/self.n_samples_per_segment
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
