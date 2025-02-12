from gampixpy.tracks import Track
from config import default_physics_params

import numpy as np
import particle

meta_dtype =  np.dtype([("event id", "u4"),
                        ("primary energy", "f4"),
                        ],
                       align = True)

class InputParser:
    def __init__(self, input_filename, sequential_sampling = True, physics_config = default_physics_params):
        self.physics_config = physics_config
        
        self.input_filename = input_filename

        self.sampling_order = []

        self.open_file_handle()
        self.generate_sample_order(sequential_sampling)
        
    def __iter__(self):
        for sample_index in self.sampling_order:
            yield sample_index, self.get_sample(sample_index), self.get_meta(sample_index)

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
    
    def get_meta(self, index):
        return None

class RooTrackerParser (InputParser):
    def get_G4_sample(self, sample_index):
        return None
        # do the magic that lets you read from a Geant4 ROOT file

    def get_sample(self, index):
        return self.get_G4_sample(index)

class QPixParser (InputParser):
    def open_file_handle(self):
        import ROOT
        self.file_handle = ROOT.TFile(self.input_filename)

        self.ttree = self.file_handle.Get("event_tree")

    def generate_sample_order(self, sequential_sampling):
        n_images_per_file = self.ttree.GetEntries()

    def get_sample(self, index):
        # from array import array
        import numpy as np

        # self.ttree.GetEntry(index)
        self.number_particles = np.array([0])
        self.hit_start_x = np.array([0])
        N = 1
        self.particle_pdg_code = np.array(N*[0])
        self.particle_mass = np.array(N*[0.])

        self.particle_initial_x = np.array(N*[0])
        self.particle_initial_y = np.array(N*[0])
        self.particle_initial_z = np.array(N*[0])

        self.ttree.SetBranchAddress("number_particles", self.number_particles)
        self.ttree.SetBranchAddress("hit_start_x", self.hit_start_x)
        self.ttree.SetBranchAddress("particle_pdg_code", self.particle_pdg_code)
        self.ttree.SetBranchAddress("particle_mass", self.particle_mass)

        self.ttree.SetBranchAddress("particle_initial_x", self.particle_initial_x)
        self.ttree.SetBranchAddress("particle_initial_y", self.particle_initial_y)
        self.ttree.SetBranchAddress("particle_initial_z", self.particle_initial_z)

        self.ttree.GetEntry(index)

        # self.particle_charge = array('f', self.number_particles*[0.])
        # self.hit_start_x = array('f', self.number_particles*[0.])
        # self.particle_charge = array('f', [0.])
        # self.hit_start_x = array('i', [0])

        # self.ttree.SetBranchAddress("particle_charge", self.particle_charge)

        self.ttree.GetEntry(index)

        print (self.number_particles,
               # self.particle_charge,
               # self.hit_start_x,
               self.particle_pdg_code,
               self.particle_mass,
               # self.particle_initial_x,
               # self.particle_initial_y,
               # self.particle_initial_z,
               )
        return None

    def get_meta(self, index):
        return None

    
class EdepSimParser (InputParser):
    # Unit conventions for edepsim inputs:
    # distance: cm
    # energy: MeV
    def open_file_handle(self):
        import h5py
        self.file_handle = h5py.File(self.input_filename)

    def generate_sample_order(self, sequential_sampling):
        unique_event_ids = np.unique(self.file_handle['trajectories']['eventID'])
        n_images_per_file = len(unique_event_ids)
        if sequential_sampling:
            self.sampling_order = unique_event_ids
        else:
            self.sampling_order = np.random.choice(unique_event_ids,
                                                   n_images_per_file,
                                                   replace = False)
        
    def get_edepsim_event(self, sample_index):
        segment_mask = self.file_handle['segments']['eventID'] == sample_index
        event_segments = self.file_handle['segments'][segment_mask]

        trajectory_mask = self.file_handle['trajectories']['eventID'] == sample_index
        event_trajectories = self.file_handle['trajectories'][trajectory_mask]

        charge_per_segment = self.do_recombination(event_segments)
        charge_points, charge_values = self.do_point_sampling(event_segments, charge_per_segment)
        
        return Track(charge_points, charge_values)
    
    def get_edepsim_meta(self, sample_index):
        trajectory_mask = self.file_handle['trajectories']['eventID'] == sample_index
        event_trajectories = self.file_handle['trajectories'][trajectory_mask]
        primary_trajectory = event_trajectories[event_trajectories['parentID'] == -1]
        print ("primary", primary_trajectory, primary_trajectory.dtype)

        pdg_code = primary_trajectory['pdgId']
        mass = particle.Particle.from_pdgid(pdg_code).mass # MeV/c^2
        momentum = primary_trajectory['pxyz_start'] # MeV/c
        kinetic_energy = np.sqrt(np.power(mass, 2) + np.sum(np.power(momentum, 2))) - mass

        meta_array = np.array([(sample_index,
                                kinetic_energy)],
                              dtype = meta_dtype)
        print ("meta array", meta_array)
        
        return meta_array
        
    def do_recombination(self, segments):
        dE = segments['dE']
        dEdx = segments['dEdx']
        E_field = self.physics_config['charge_drift']['drift_field']
        LAr_density = self.physics_config['material']['density']

        mode = 'box'
        # mode = 'birks'
        if mode == 'box':
            box_beta = self.physics_config['box_model']['box_alpha']
            box_alpha = self.physics_config['box_model']['box_beta']

            csi = box_beta*dEdx/(E_field*LAr_density)
            recomb = np.max(np.stack([np.zeros_like(dE),
                                      np.log(box_alpha + csi)/csi]),
                            axis = 0)

        elif mode == 'birks':
            birks_ab = self.physics_config['birks_model']['birks_ab']
            birks_kb = self.physics_config['birks_model']['birks_kb']

            recomb = birks_ab/(1 + birks_kb*dEdx/(E_field * LAr_density))

        else:
            raise ValueError("Invalid recombination mode: must be 'physics.BOX' or 'physics.BRIKS'")

        if np.any(np.isnan(recomb)):
            raise RuntimeError("Invalid recombination value")

        w_ion = self.physics_config['material']['w']
        charge_yield_per_energy = recomb/w_ion

        n_electrons = segments['dE']*charge_yield_per_energy
        return n_electrons
    
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

    def get_meta(self, index):
        return self.get_edepsim_meta(index)


class MarleyParser (InputParser):
    # Unit conventions for Marley inputs:
    # distance: UNKNOWN
    # energy: UNKNOWN
    def open_file_handle(self):
        import h5py
        self.file_handle = h5py.File(self.input_filename)

    def generate_sample_order(self, sequential_sampling):
        unique_event_ids = np.unique(self.file_handle['trajectories']['eventID'])
        n_images_per_file = len(unique_event_ids)
        if sequential_sampling:
            self.sampling_order = unique_event_ids
        else:
            self.sampling_order = np.random.choice(unique_event_ids,
                                                   n_images_per_file,
                                                   replace = False)
        
    def get_edepsim_event(self, sample_index):
        segment_mask = self.file_handle['segments']['eventID'] == sample_index
        event_segments = self.file_handle['segments'][segment_mask]

        trajectory_mask = self.file_handle['trajectories']['eventID'] == sample_index
        event_trajectories = self.file_handle['trajectories'][trajectory_mask]

        charge_per_segment = self.do_recombination(event_segments)
        charge_points, charge_values = self.do_point_sampling(event_segments, charge_per_segment)
        
        return Track(charge_points, charge_values)

    def get_edepsim_meta(self, sample_index):
        trajectory_mask = self.file_handle['trajectories']['eventID'] == sample_index
        event_trajectories = self.file_handle['trajectories'][trajectory_mask]
        primary_trajectory = event_trajectories[event_trajectories['parentID'] == -1]
        print ("primary", primary_trajectory)
        
        return None
        # return Track(charge_points, charge_values)

    def do_recombination(self, segments):
        dE = segments['dE']
        dEdx = segments['dEdx']
        E_field = self.physics_config['charge_drift']['drift_field']
        LAr_density = self.physics_config['material']['density']

        mode = 'box'
        # mode = 'birks'
        if mode == 'box':
            box_beta = self.physics_config['box_model']['box_alpha']
            box_alpha = self.physics_config['box_model']['box_beta']

            csi = box_beta*dEdx/(E_field*LAr_density)
            recomb = np.max(np.stack([np.zeros_like(dE),
                                      np.log(box_alpha + csi)/csi]),
                            axis = 0)

        elif mode == 'birks':
            birks_ab = self.physics_config['birks_model']['birks_ab']
            birks_kb = self.physics_config['birks_model']['birks_kb']

            recomb = birks_ab/(1 + birks_kb*dEdx/(E_field * LAr_density))

        else:
            raise ValueError("Invalid recombination mode: must be 'physics.BOX' or 'physics.BRIKS'")

        if np.any(np.isnan(recomb)):
            raise RuntimeError("Invalid recombination value")

        w_ion = self.physics_config['material']['w']
        charge_yield_per_energy = recomb/w_ion

        n_electrons = segments['dE']*charge_yield_per_energy
        return n_electrons
    
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

    def get_meta(self, index):
        print ("calling get_meta method")
        return self.get_edepsim_meta(index)
