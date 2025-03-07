from gampixpy.tracks import Track
from gampixpy.config import default_physics_params

import numpy as np
import particle

meta_dtype =  np.dtype([("event id", "u4"),
                        ("primary energy", "f4"),
                        ("deposited charge", "f4"),
                        ("vertex x", "f4"),
                        ("vertex y", "f4"),
                        ("vertex z", "f4"),
                        ("theta", "f4"),
                        ("phi", "f4"),
                        ("primary length", "f4"),
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

class RooTrackerParser (InputParser):
    def open_file_handle(self):
        from ROOT import TFile, TG4Event

        self.file_handle = TFile(self.input_filename)
        self.inputTree = self.file_handle.Get("EDepSimEvents")

        self.event = TG4Event()
        self.inputTree.SetBranchAddress("Event", self.event)

    def generate_sample_order(self, sequential_sampling):
        n_images_per_file = self.inputTree.GetEntriesFast()
        if sequential_sampling:
            self.sampling_order = np.arange(n_images_per_file)
        else:
            self.sampling_order = np.random.choice(n_images_per_file,
                                                   n_images_per_file,
                                                   replace = False)

    def get_G4_sample(self, sample_index):
        self.inputTree.GetEntry(sample_index)

        segment_dtype = np.dtype([("x_start", "f4"),
                                  ("y_start", "f4"),
                                  ("z_start", "f4"),
                                  ("x_end", "f4"),
                                  ("y_end", "f4"),
                                  ("z_end", "f4"),
                                  ("dE", "f4"),
                                  ("dx", "f4"),
                                  ("dEdx", "f4")],
                                 align = True)
        segment_array = np.array([], dtype = segment_dtype)

        for container_name, hit_segments in self.event.SegmentDetectors:
            for segment in hit_segments:
                x_start = segment.GetStart().X()
                y_start = segment.GetStart().Y()
                z_start = segment.GetStart().Z()
                x_end = segment.GetStop().X()
                y_end = segment.GetStop().Y()
                z_end = segment.GetStop().Z()

                x_d = x_end - x_start
                y_d = y_end - y_start
                z_d = z_end - z_start
                dx = np.sqrt(x_d**2 + y_d**2 + z_d**2)
                
                dE = segment.GetEnergyDeposit()
                dEdx = dE/dx if dx > 0 else 0
                
                this_segment_array = np.array([(x_start,
                                                y_start,
                                                z_start,
                                                x_end,
                                                y_end,
                                                z_end,
                                                dE,
                                                dx,
                                                dEdx)],
                                              dtype = segment_dtype)
                segment_array = np.concatenate((this_segment_array,
                                                segment_array))

        charge_per_segment = self.do_recombination(segment_array)
        charge_points, charge_values = self.do_point_sampling(segment_array, charge_per_segment)

        return Track(charge_points, charge_values)

    def get_G4_meta(self, sample_index):
        primary_vertex = self.event.Primaries[0] # assume only one primary for now

        vertex_x = primary_vertex.GetPosition().X()
        vertex_y = primary_vertex.GetPosition().Y()
        vertex_z = primary_vertex.GetPosition().Z()
        
        primary_trajectory = self.event.Trajectories[0]
        assert primary_trajectory.GetParentId() == -1

        pdg_code = primary_trajectory.GetPDGCode()
        mass = particle.Particle.from_pdgid(pdg_code).mass # MeV/c^2
        energy = primary_trajectory.GetInitialMomentum()[3]
        momentum = [primary_trajectory.GetInitialMomentum()[i] for i in range(3)]
        kinetic_energy = energy - mass

        theta = np.arctan2(momentum[1], momentum[0])
        phi = np.arctan2(np.sqrt(momentum[0]**2 + momentum[1]**2), momentum[2])
        
        meta_array = np.array([(sample_index,
                                kinetic_energy,
                                -1, # charge undefined
                                vertex_x, vertex_y, vertex_z,
                                theta, phi,
                                -1, # primary length undefined
                                )],
                              dtype = meta_dtype)
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
        return self.get_G4_sample(index)

    def get_meta(self, index):
        return self.get_G4_meta(index)

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

        pdg_code = primary_trajectory['pdgId']
        mass = particle.Particle.from_pdgid(pdg_code).mass # MeV/c^2
        momentum = primary_trajectory['pxyz_start'] # MeV/c
        kinetic_energy = np.sqrt(np.power(mass, 2) + np.sum(np.power(momentum, 2))) - mass

        vertex = primary_trajectory['xyz_start'][0]
        init_momentum = primary_trajectory['pxyz_start'][0]

        theta = np.arctan2(init_momentum[1], init_momentum[0])
        phi = np.arctan2(np.sqrt(init_momentum[0]**2 + init_momentum[1]**2), init_momentum[2])
        
        meta_array = np.array([(sample_index,
                                kinetic_energy,
                                -1, # charge undefined
                                vertex[0], vertex[1], vertex[2],
                                theta, phi,
                                -1, # primary length undefined
                                )],
                              dtype = meta_dtype)
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
    def open_file_handle(self):
        from ROOT import TFile

        self.file_handle = TFile(self.input_filename)
        self.inputTree = self.file_handle.Get("edep")

    def generate_sample_order(self, sequential_sampling):
        event_ids = []
        for entry in self.inputTree:
            event_ids.append(entry.event)
        unique_event_ids = np.unique(event_ids)
        n_images_per_file = self.inputTree.GetEntriesFast()
        if sequential_sampling:
            self.sampling_order = unique_event_ids
        else:
            self.sampling_order = np.random.choice(unique_event_ids,
                                                   len(unique_event_ids),
                                                   replace = False)
    
    def get_G4_sample(self, sample_index):
        segment_dtype = np.dtype([("x_start", "f4"),
                                  ("y_start", "f4"),
                                  ("z_start", "f4"),
                                  ("x_end", "f4"),
                                  ("y_end", "f4"),
                                  ("z_end", "f4"),
                                  ("dE", "f4"),
                                  ("dx", "f4"),
                                  ("dEdx", "f4")],
                                align = True)
        segment_array = np.array([], dtype = segment_dtype)

        for entry in self.inputTree:
            if entry.event == sample_index:
                x_start = entry.startX
                y_start = entry.startY
                z_start = entry.startZ
                x_end = entry.endX
                y_end = entry.endY
                z_end = entry.endZ

                x_d = x_end - x_start
                y_d = y_end - y_start
                z_d = z_end - z_start
                dx = np.sqrt(x_d**2 + y_d**2 + z_d**2)
                
                dE = entry.dE
                dEdx = dE/dx if dx > 0 else 0

                this_segment_array = np.array([(x_start,
                                                y_start,
                                                z_start,
                                                x_end,
                                                y_end,
                                                z_end,
                                                dE,
                                                dx,
                                                dEdx)],
                                              dtype = segment_dtype)
                segment_array = np.concatenate((this_segment_array,
                                                segment_array))

        charge_per_segment = self.do_recombination(segment_array)
        charge_points, charge_values = self.do_point_sampling(segment_array, charge_per_segment)

        return Track(charge_points, charge_values)

    def get_G4_meta(self, sample_index):
        meta_array = np.array([(sample_index,
                                -1, # KE undefined
                                -1, # charge undefined
                                -1, -1, -1, # vertex point undefined
                                -1, -1, # primary attitude undefined
                                -1, # primary length undefined
                                )],
                              dtype = meta_dtype)
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
        return self.get_G4_sample(index)

    def get_meta(self, index):
        return self.get_G4_meta(index)

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

