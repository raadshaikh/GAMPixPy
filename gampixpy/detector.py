from config import default_detector_params, default_physics_params, default_readout_params
from readout_objects import PixelSample, CoarseGridSample

import numpy as np

class GAMPixModel:
    def __init__(self, readout_config = default_readout_params):
        self.readout_config = readout_config

    def electronics_simulation(self, track):
        # apply the electronics response simulation to
        # a track containing a drifted charge sample

        print ("simulating coarse grid...")
        # do coarse grid binning/time series formation
        coarse_tile_timeseries = self.transverse_tile_binning(track)        
        # find hits on coarse grid
        self.coarse_tile_hits = self.tile_hit_finding(track, coarse_tile_timeseries)

        print ("simulating fine grid...")
        fine_pixel_timeseries = self.transverse_pixel_binning(track, self.coarse_tile_hits)
        # find hits on fine pixels
        self.fine_pixel_hits = self.pixel_hit_finding(track, fine_pixel_timeseries)

        return 

    def transverse_tile_binning(self, track):
        min_tile = np.array([self.readout_config['anode']['x_range'][0],
                             self.readout_config['anode']['y_range'][0],
                             ])
        spacing = self.readout_config['coarse_tiles']['pitch']
        tile_ind = np.asarray((track.drifted_track['position'][:,[0, 1]] - min_tile)//spacing, dtype = int)

        inside_anode_mask = (np.min(tile_ind, axis = -1) >= 0) 
        inside_anode_mask *= tile_ind[:, 0] < self.readout_config['n_tiles_x'] 
        inside_anode_mask *= tile_ind[:, 1] < self.readout_config['n_tiles_y']

        tile_ind = tile_ind[inside_anode_mask]

        tile_centers = np.array([self.readout_config['tile_volume_edges'][i][tile_ind[:,i]] + 0.5*self.readout_config['coarse_tiles']['pitch']
                                 for i in range(2)]).T

        z_series = track.drifted_track['position'][inside_anode_mask,2]
        charge_series = track.drifted_track['charge'][inside_anode_mask]
        
        coarse_grid_timeseries = {}
        for tile_center, drift_position, charge in zip(tile_centers, z_series, charge_series):
            # number of rounding digits is potentially problematic 
            tile_coord = (round(float(tile_center[0]), 3),
                          round(float(tile_center[1]), 3))
            if tile_coord in coarse_grid_timeseries:
                coarse_grid_timeseries[tile_coord] = np.concatenate((coarse_grid_timeseries[tile_coord],
                                                                     np.array([[drift_position, charge]])),
                                                                    axis = 0)
            else:
                coarse_grid_timeseries[tile_coord] = np.array([[drift_position, charge]])

        return coarse_grid_timeseries

    def tile_hit_finding(self, track, tile_timeseries, method = 'charge_density'):
        if method == 'current_rise':
            return self.tile_hit_finding_current_rise(track, tile_timeseries)
        elif method == 'charge_density':
            return self.tile_hit_finding_charge_density(track, tile_timeseries)

    def tile_hit_finding_charge_density(self, track, tile_timeseries):
        """
        This method of hit finding simply looks for a quantity
        of charge above threshold within a given z-bin
        (corresponding to a clock_period*integration_length)
        """
        # TODO: fix logic for integration_length > 1
        hits = []

        # set up the clock cycle boundaries
        n_z_bins = int((self.readout_config['anode']['z_range'][1] - self.readout_config['anode']['z_range'][0])/self.readout_config['coarse_tiles']['z_bin_width'])
        drift_bin_edges = np.linspace(self.readout_config['anode']['z_range'][0],
                                      self.readout_config['anode']['z_range'][1],
                                      n_z_bins + 1,
                                      )
        
        for tile_center, timeseries in tile_timeseries.items():
            # find the charge which falls into each clock bin
            inst_charge, _ = np.histogram(timeseries[:,0],
                                          weights = timeseries[:,1],
                                          bins = drift_bin_edges)
            # cumulative charge since last digitized sample
            cum_charge = np.cumsum(inst_charge)
            cum_charge = np.concatenate(([0], cum_charge))

            # search along the bins until no more threshold crossings
            no_more_hits = False
            while not no_more_hits:
                # take a diff to find the amount of charge within a given window
                # window size is (clock period)*(integration length)
                padded_charge = np.pad(cum_charge,
                                       self.readout_config['coarse_tiles']['integration_length'],
                                       mode = 'edge')
                window_charge = padded_charge[self.readout_config['coarse_tiles']['integration_length']:] -\
                                padded_charge[:-self.readout_config['coarse_tiles']['integration_length']]

                threshold = self.readout_config['coarse_tiles']['noise']*self.readout_config['coarse_tiles']['threshold_sigma']

                threshold_crossing_mask = window_charge[1:-self.readout_config['coarse_tiles']['integration_length']] > threshold

                if np.any(threshold_crossing_mask):
                    threshold_crossing_z = drift_bin_edges[:-1][threshold_crossing_mask][0]
                    threshold_crossing_charge = window_charge[1:-self.readout_config['coarse_tiles']['integration_length']][threshold_crossing_mask][0]

                    cum_charge = cum_charge - threshold_crossing_charge
                    cum_charge = np.max(np.stack([cum_charge, np.zeros_like(cum_charge)]), axis = 0) # don't subtract charge below zero
            
                    hits.append(CoarseGridSample(tile_center,
                                                 threshold_crossing_z,
                                                 threshold_crossing_charge))
                else:
                    no_more_hits = True

        track.coarse_tiles_samples = hits
        return hits 

    def transverse_pixel_binning(self, track, coarse_tile_hits):
        fine_pixel_timeseries = {}

        spacing = self.readout_config['pixels']['pitch']
        
        for this_coarse_hit in coarse_tile_hits:

            cell_center_xy = this_coarse_hit.coarse_cell_id # may change
            cell_edge_z = this_coarse_hit.coarse_measurement_time
            
            x_bounds = [cell_center_xy[0] - 0.5*self.readout_config['coarse_tiles']['pitch'],
                        cell_center_xy[0] + 0.5*self.readout_config['coarse_tiles']['pitch']]
            y_bounds = [cell_center_xy[1] - 0.5*self.readout_config['coarse_tiles']['pitch'],
                        cell_center_xy[1] + 0.5*self.readout_config['coarse_tiles']['pitch']]
            z_bounds = [cell_edge_z,
                        cell_edge_z + self.readout_config['coarse_tiles']['z_bin_width']*self.readout_config['coarse_tiles']['integration_length']]

            in_cell_mask = track.drifted_track['position'][:,0] >= x_bounds[0]
            in_cell_mask *= track.drifted_track['position'][:,0] < x_bounds[1]
            in_cell_mask *= track.drifted_track['position'][:,1] >= y_bounds[0]
            in_cell_mask *= track.drifted_track['position'][:,1] < y_bounds[1]

            in_cell_positions = track.drifted_track['position'][in_cell_mask]
            in_cell_charges = track.drifted_track['charge'][in_cell_mask]
            
            min_pixel = np.array([x_bounds[0],
                                  y_bounds[0],
                                  ])
            tile_ind = np.asarray((in_cell_positions[:,[0, 1]] - min_pixel)//spacing, dtype = int)

            n_pixels_x = int((x_bounds[1] - x_bounds[0])/spacing)
            n_pixels_y = int((y_bounds[1] - y_bounds[0])/spacing)
            pixel_volume_edges = (np.linspace(x_bounds[0], x_bounds[1], n_pixels_x+1),
                                  np.linspace(y_bounds[0], y_bounds[1], n_pixels_y+1))
            pixel_centers = np.array([pixel_volume_edges[i][tile_ind[:,i]] + 0.5*spacing
                                      for i in range(2)]).T
            
            z_series = in_cell_positions[:,2]
            charge_series = in_cell_charges
        
            for pixel_center, drift_position, charge in zip(pixel_centers, z_series, charge_series):
                # number of rounding digits is potentially problematic 
                pixel_coord = (round(float(pixel_center[0]), 3),
                               round(float(pixel_center[1]), 3))
                if pixel_coord in fine_pixel_timeseries:
                    fine_pixel_timeseries[pixel_coord] = np.concatenate((fine_pixel_timeseries[pixel_coord],
                                                                         np.array([[drift_position, charge]])),
                                                                        axis = 0)
                else:
                    fine_pixel_timeseries[pixel_coord] = np.array([[drift_position, charge]])

        return fine_pixel_timeseries

    def pixel_hit_finding(self, track, pixel_timeseries):
        """
        This method of hit finding simply looks for a quantity
        of charge above threshold within a given z-bin
        (corresponding to a clock_period*integration_length)
        """
        # TODO: fix logic for integration_length > 1
        # TODO: actually constrain hit finding to within coarse grid volume (might be implicitly true)
        hits = []

        # set up the clock cycle boundaries
        n_z_bins = int((self.readout_config['anode']['z_range'][1] - self.readout_config['anode']['z_range'][0])/self.readout_config['pixels']['z_bin_width'])
        drift_bin_edges = np.linspace(self.readout_config['anode']['z_range'][0],
                                      self.readout_config['anode']['z_range'][1],
                                      n_z_bins + 1,
                                      )
        
        for pixel_center, timeseries in pixel_timeseries.items():
            # find the charge which falls into each clock bin
            inst_charge, _ = np.histogram(timeseries[:,0],
                                          weights = timeseries[:,1],
                                          bins = drift_bin_edges)
            # cumulative charge since last digitized sample
            cum_charge = np.cumsum(inst_charge)
            cum_charge = np.concatenate(([0], cum_charge))

            # search along the bins until no more threshold crossings
            no_hits = False
            while not no_hits:
                # take a diff to find the amount of charge within a given window
                # window size is (clock period)*(integration length)
                padded_charge = np.pad(cum_charge,
                                       self.readout_config['pixels']['integration_length'],
                                       mode = 'edge')
                window_charge = padded_charge[self.readout_config['pixels']['integration_length']:] -\
                                padded_charge[:-self.readout_config['pixels']['integration_length']]

                # threshold = 5.e-5 # not really the hit threshold
                threshold = self.readout_config['pixels']['noise']*self.readout_config['pixels']['threshold_sigma']
                threshold_crossing_mask = window_charge[1:-self.readout_config['pixels']['integration_length']] > threshold

                if np.any(threshold_crossing_mask):
                    threshold_crossing_z = drift_bin_edges[:-1][threshold_crossing_mask][0]
                    threshold_crossing_charge = window_charge[1:-self.readout_config['pixels']['integration_length']][threshold_crossing_mask][0]

                    cum_charge = cum_charge - threshold_crossing_charge
                    cum_charge = np.max(np.stack([cum_charge, np.zeros_like(cum_charge)]), axis = 0) # don't subtract charge below zero
            
                    hits.append(PixelSample(pixel_center,
                                            threshold_crossing_z,
                                            threshold_crossing_charge))
                else:
                    no_hits = True

        track.pixel_samples = hits
        return hits 

class DetectorModel:
    def __init__(self,
                 detector_params = default_detector_params,
                 physics_params = default_physics_params,
                 readout_params = default_readout_params):
        self.detector_params = detector_params
        self.physics_params = physics_params
        self.readout_params = readout_params
        self.readout_model = GAMPixModel(readout_params)
 
    def simulate(self, track):
        self.drift(track)
        self.readout(track)
        
    def drift(self, sampled_track):
        # drift the charge samples from their input position
        # to the anode position as defined by detector_params
        # save the drifted positions to te track

        # TODO: a more complete way to describe the anode geometry
        # i.e., specify a plane and assume drift direction is shortest
        # path to that plane
        # anode_z = self.detector_params['anode']['z']
        anode_z = self.readout_params['anode']['z_range'][0]

        input_position = sampled_track.raw_track['position']
        input_charges = sampled_track.raw_track['charge']

        # position is disturbed by diffusion
        drift_distance = input_position[:,2] - anode_z

        # mask all points which are behind the anode
        region_mask = drift_distance > 0
        region_position = input_position[region_mask]
        region_charges = input_charges[region_mask]

        drift_distance = drift_distance[region_mask]

        # TODO: implement better velocity model (now set as a constant in params)
        drift_time = drift_distance/self.physics_params['charge_drift']['drift_speed'] # s
        # TODO: better diffusion model
        # D accordint to https://lar.bnl.gov/properties/trans.html#diffusion-l
        # sigma = sqrt(2*D*t)
        # diffusion_sigma = np.array([self.physics_params['spatial_resolution']['sigma_transverse'],
        #                             self.physics_params['spatial_resolution']['sigma_transverse'],
        #                             self.physics_params['spatial_resolution']['sigma_longitudinal'],
        #                             ])
        sigma_transverse = np.sqrt(2*self.physics_params['charge_drift']['diffusion_transverse']*drift_time)
        sigma_longitudinal = np.sqrt(2*self.physics_params['charge_drift']['diffusion_longitudinal']*drift_time)
        # sigma_transverse = 0
        # sigma_longitudinal = 0
        diffusion_sigma = np.array([sigma_transverse,
                                    sigma_transverse,
                                    sigma_longitudinal,
                                    ]).T
        
        drifted_positions = np.random.normal(loc = region_position,
                                             scale = diffusion_sigma*np.ones_like(region_position))

        # charge is disturbed by attenuation
        drifted_charges = region_charges*np.exp(-drift_time/self.physics_params['charge_drift']['electron_lifetime'])

        # might also include a sub-sampling step?
        # in case initial sampling is not fine enough

        sampled_track.drifted_track = {'position': drifted_positions,
                                       'charge': drifted_charges}
        return 

    def readout(self, drifted_track):
        # apply the readout simulation to the drifted track
        # this is defined by the GAMPixModel object

        self.readout_model.electronics_simulation(drifted_track)

        return
