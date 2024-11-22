from .tracks import Track

import numpy as np

class InputParser:
    def __init__(self, input_filename, sequential_sampling = True):
        self.input_filename = input_filename

        self.sampling_order = []

        self.open_file_handle()
        self.generate_sample_order(sequential_sampling)
        
    def __iter__(self):
        for sample_index in self.sampling_order:
            yield self.get_sample(sample_index)

class PenelopeParser (InputParser):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    
    def open_file_handle(self):
        # import h5py
        # self.file_handle = h5py.File(self.input_filename)
        return 

    def generate_sample_order(self, sequential_sampling):
        # n_images_per_file = len(np.unique(self.file_handle['trajectories']['eventID']))
        return 
        
    def get_penelope_sample(self):
        print ("wowzers")
        print (self.input_filename)
        # do the magic that lets you read from a penelope file
        return None

    def get_sample(self, index):
        return self.get_penelope_sample(index)

class RooTrackerParser (InputParser):
    def get_G4_sample(self, sample_index):
        return None
        # do the magic that lets you read from a Geant4 ROOT file

    def get_sample(self, index):
        return self.get_G4_sample(index)

class EdepSimParser (InputParser):
    def open_file_handle(self):
        import h5py
        self.file_handle = h5py.File(self.input_filename)

    def generate_sample_order(self, sequential_sampling):
        n_images_per_file = len(np.unique(self.file_handle['trajectories']['eventID']))
        
    def get_edepsim_event(self, sample_index):
        segment_mask = self.file_handle['segments']['eventID'] == sample_index
        event_segments = self.file_handle['segments'][segment_mask]

        trajectory_mask = self.file_handle['trajectories']['eventID'] == sample_index
        event_trajectories = self.file_handle['trajectories'][trajectory_mask]

        charge_per_segment = self.do_recombination(event_segments)
        charge_points, charge_values = self.do_point_sampling(event_segments, charge_per_segment)
        
        return Track(charge_points, charge_values)

    def do_recombination(self, segments):
        charge_yield_per_energy = 1 # just do a constant factor right now
    
        return segments['dE']*charge_yield_per_energy
    
    def do_point_sampling(self, segments, charge_per_segment):
        # point sampling with a fixed number of samples per length
        # it may be faster to do sampling another way (test in future!)
        #  - sample with fixed amount of charge
        #  - sample with fixed number of samples per segment

        # sample_density = 1.e4 # samples per unit length
        sample_density = 1.e3 # samples per unit length
        start_vec = np.array([segments['x_start'],
                              segments['y_start'],
                              segments['z_start']])
        end_vec = np.array([segments['x_end'],
                            segments['y_end'],
                            segments['z_end']])
        segment_dirs = end_vec - start_vec

        samples_per_segment = (segments['dx']*sample_density).astype(int)

        sample_positions = np.concatenate([(start_vec[:,i,None] + np.linspace(0, 1, samples_per_segment[i])*segment_dirs[:,i,None]).T
                                           for i in range(len(samples_per_segment))])

        sample_charges = np.concatenate([samples_per_segment[i]*[charge_per_segment[i]/samples_per_segment[i]]
                                         for i in range(len(samples_per_segment))])
        
        return sample_positions, sample_charges

    def get_sample(self, index):
        return self.get_edepsim_event(index)
