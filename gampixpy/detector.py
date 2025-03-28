from gampixpy.config import default_detector_params, default_physics_params, default_readout_params
from gampixpy.readout_objects import PixelSample, CoarseGridSample

import numpy as np

class GAMPixModel:
    def __init__(self, readout_config = default_readout_params):
        self.readout_config = readout_config

    def electronics_simulation(self, track, verbose = True, **kwargs):
        # apply the electronics response simulation to
        # a track containing a drifted charge sample

        self.clock_start_time = np.min(track.drifted_track['times'])

        if verbose:
            print ("simulating coarse grid...")
        # do coarse grid binning/time series formation
        coarse_tile_timeseries = self.transverse_tile_binning(track, **kwargs)
        if verbose:
            print ("coarse time series built")
            print ("coarse hit finding...")
        # find hits on coarse grid
        self.coarse_tile_hits = self.tile_hit_finding(track, coarse_tile_timeseries, **kwargs)

        if verbose:
            print ("simulating fine grid...")
        fine_pixel_timeseries = self.transverse_pixel_binning(track, self.coarse_tile_hits, **kwargs)
        if verbose:
            print ("pixel time series built")
            print ("pixel hit finding...")
        # find hits on fine pixels
        self.fine_pixel_hits = self.pixel_hit_finding(track, fine_pixel_timeseries, **kwargs)

        return 

    def transverse_tile_binning(self, track, **kwargs):
        min_tile = np.array([self.readout_config['anode']['x_range'][0],
                             self.readout_config['anode']['y_range'][0],
                             ])
        spacing = self.readout_config['coarse_tiles']['pitch']
        tile_ind = np.asarray((track.drifted_track['position'][:,[0, 1]] - min_tile)//spacing, dtype = int)

        inside_anode_mask = (np.min(tile_ind, axis = -1) >= 0) 
        inside_anode_mask *= tile_ind[:, 0] < self.readout_config['n_tiles_x'] 
        inside_anode_mask *= tile_ind[:, 1] < self.readout_config['n_tiles_y']

        tile_ind = tile_ind[inside_anode_mask]
        tile_hash = np.array([hash(tuple(ind)) for ind in tile_ind])

        z_series = track.drifted_track['position'][inside_anode_mask,2]
        t_series = track.drifted_track['times'][inside_anode_mask]
        charge_series = track.drifted_track['charge'][inside_anode_mask]
        
        coarse_grid_timeseries = {}
        unique_tile_hashes, unique_tile_key = np.unique(tile_hash, return_index = True)
        unique_tile_indices = tile_ind[unique_tile_key]
        
        for this_tile_hash, this_tile_ind in zip(unique_tile_hashes, unique_tile_indices):
            tile_center = np.array([self.readout_config['tile_volume_edges'][i][this_tile_ind[i]] + 0.5*self.readout_config['coarse_tiles']['pitch']
                                    for i in range(2)]).T
            tile_coord = (round(float(tile_center[0]), 3),
                          round(float(tile_center[1]), 3))
            
            sample_mask = tile_hash == this_tile_hash
            tile_hit_drift_positions = z_series[sample_mask]
            tile_hit_arrival_times = t_series[sample_mask]
            tile_hit_charges = charge_series[sample_mask]

            coarse_grid_timeseries[tile_coord] = np.array([tile_hit_arrival_times,
                                                           tile_hit_drift_positions,
                                                           tile_hit_charges]).T


        return coarse_grid_timeseries

    def tile_hit_finding(self, track, tile_timeseries, nonoise = False, **kwargs):
        """
        This method of hit finding simply looks for a quantity
        of charge above threshold within a given z-bin
        (corresponding to a clock_period*integration_length)
        """
        # TODO: fix logic for integration_length > 1
        hits = []
        
        for tile_center, timeseries in tile_timeseries.items():
            # set up the clock cycle boundaries
            # binning using arrival time (allows for asynchronous ionization)
            last_charge_arrival_time = np.max(timeseries[:,0])
            n_clock_ticks = int(np.ceil((last_charge_arrival_time - self.clock_start_time)/self.readout_config['coarse_tiles']['clock_interval']))
            # this is the densest possible set of bins
            # making a dense histogram of these is going to be too much data
            # could do a hierarchical search, go from coarse to fine time binning
            # maybe using np.digitize
            arrival_time_bin_edges = np.linspace(self.clock_start_time,
                                                 self.clock_start_time + n_clock_ticks*self.readout_config['coarse_tiles']['clock_interval'],
                                                 n_clock_ticks + 1,
                                                 )

            # find the charge which falls into each clock bin
            inst_charge, _ = np.histogram(timeseries[:,0],
                                          weights = timeseries[:,2],
                                          bins = arrival_time_bin_edges)
            
            hold_length = self.readout_config['coarse_tiles']['integration_length']            
            
            # search along the bins until no more threshold crossings
            no_more_hits = False
            while not no_more_hits:
                window_charge = np.convolve(inst_charge,
                                            np.ones(hold_length))
                window_charge = window_charge[hold_length-1:]
                
                threshold = self.readout_config['coarse_tiles']['noise']*self.readout_config['coarse_tiles']['threshold_sigma']

                threshold_crossing_mask = window_charge > threshold
                threshold_crossing_mask *= inst_charge > 0

                if np.any(threshold_crossing_mask):
                    hit_index = np.where(threshold_crossing_mask)[0][0]
                    
                    threshold_crossing_t = arrival_time_bin_edges[:-1][hit_index]
                    threshold_crossing_z = threshold_crossing_t*1.6e5 # is there a better way to do this?
                    # TODO: also, get that from physics
                    
                    threshold_crossing_charge = window_charge[hit_index]
                    if not nonoise:
                        threshold_crossing_charge += np.random.normal(scale = self.readout_config['coarse_tiles']['noise'])

                    inst_charge[:hit_index+hold_length] = 0

                    hits.append(CoarseGridSample(tile_center,
                                                 threshold_crossing_t,
                                                 threshold_crossing_z,
                                                 threshold_crossing_charge))
                else:
                    no_more_hits = True

        track.coarse_tiles_samples = hits
        return hits 

    def transverse_pixel_binning(self, track, coarse_tile_hits, **kwargs):
        fine_pixel_timeseries = {}

        spacing = self.readout_config['pixels']['pitch']
        
        for this_coarse_hit in coarse_tile_hits:

            cell_center_xy = this_coarse_hit.coarse_cell_pos # may change
            cell_trigger_t = this_coarse_hit.coarse_measurement_time
            
            x_bounds = [cell_center_xy[0] - 0.5*self.readout_config['coarse_tiles']['pitch'],
                        cell_center_xy[0] + 0.5*self.readout_config['coarse_tiles']['pitch']]
            y_bounds = [cell_center_xy[1] - 0.5*self.readout_config['coarse_tiles']['pitch'],
                        cell_center_xy[1] + 0.5*self.readout_config['coarse_tiles']['pitch']]
            # z_bounds = [cell_edge_z,
            #             cell_edge_z + self.readout_config['coarse_tiles']['z_bin_width']*self.readout_config['coarse_tiles']['integration_length']]
            t_bounds = [cell_trigger_t,
                        cell_trigger_t + self.readout_config['coarse_tiles']['clock_interval']*self.readout_config['coarse_tiles']['integration_length']]

            in_cell_mask = track.drifted_track['position'][:,0] >= x_bounds[0]
            in_cell_mask *= track.drifted_track['position'][:,0] < x_bounds[1]
            in_cell_mask *= track.drifted_track['position'][:,1] >= y_bounds[0]
            in_cell_mask *= track.drifted_track['position'][:,1] < y_bounds[1]
            in_cell_mask *= track.drifted_track['times'] >= t_bounds[0]
            in_cell_mask *= track.drifted_track['times'] < t_bounds[1]

            in_cell_positions = track.drifted_track['position'][in_cell_mask]
            in_cell_charges = track.drifted_track['charge'][in_cell_mask]
            
            min_pixel = np.array([x_bounds[0],
                                  y_bounds[0],
                                  ])
            n_pixels_x = int((x_bounds[1] - x_bounds[0])/spacing)
            n_pixels_y = int((y_bounds[1] - y_bounds[0])/spacing)
            pixel_volume_edges = (np.linspace(x_bounds[0], x_bounds[1], n_pixels_x+1),
                                  np.linspace(y_bounds[0], y_bounds[1], n_pixels_y+1))
            
            z_series = in_cell_positions[:,2]
            t_series = track.drifted_track['times'][in_cell_mask]
            charge_series = in_cell_charges

            # generate a unique id for each pixel within this coarse hit
            # so that hits from 
            pixel_ind = np.asarray((in_cell_positions[:,[0, 1]] - min_pixel)//spacing, dtype = int)
            pixel_hash = np.array([hash(tuple(ind)+tuple(cell_center_xy)) for ind in pixel_ind])

            unique_pixel_hashes, unique_pixel_key = np.unique(pixel_hash, return_index = True)
            unique_pixel_indices = pixel_ind[unique_pixel_key]

            for this_pixel_hash, this_pixel_ind in zip(unique_pixel_hashes, unique_pixel_indices):
                pixel_center = np.array([pixel_volume_edges[i][this_pixel_ind[i]] + 0.5*spacing
                                         for i in range(2)]).T
                pixel_coord = (round(float(pixel_center[0]), 3),
                               round(float(pixel_center[1]), 3))

                sample_mask = pixel_hash == this_pixel_hash

                this_hit_drift_positions = z_series[sample_mask]
                this_hit_arrival_times = t_series[sample_mask]
                this_hit_charges = charge_series[sample_mask]
                fine_pixel_timeseries[this_pixel_hash] = {"coarse hit": this_coarse_hit,
                                                          "pixel coord": pixel_coord,
                                                          "time series": np.array([this_hit_arrival_times,
                                                                                   this_hit_drift_positions,
                                                                                   this_hit_charges]).T,
                                                          }
                                           
        return fine_pixel_timeseries

    def pixel_hit_finding(self, track, pixel_timeseries, nonoise = False, **kwargs):
        """
        This method of hit finding simply looks for a quantity
        of charge above threshold within a given z-bin
        (corresponding to a clock_period*integration_length)
        """
        hits = []
        
        for pixel_dict in pixel_timeseries.values():
            coarse_hit = pixel_dict['coarse hit']
            pixel_center = pixel_dict['pixel coord']
            timeseries = pixel_dict['time series']
            # find the charge which falls into each clock bin
            last_charge_arrival_time = np.max(timeseries[:,0])
            # use the tile hit time instead of global start time
            n_clock_ticks = int(np.ceil((last_charge_arrival_time - coarse_hit.coarse_measurement_time)/self.readout_config['pixels']['clock_interval']))
            # this is the densest possible set of bins
            # making a dense histogram of these is going to be too much data
            # could do a hierarchical search, go from coarse to fine time binning
            # maybe using np.digitize
            arrival_time_bin_edges = np.linspace(coarse_hit.coarse_measurement_time,
                                                 coarse_hit.coarse_measurement_time + n_clock_ticks*self.readout_config['pixels']['clock_interval'],
                                                 n_clock_ticks + 1,
                                                 )
            
            # find the charge which falls into each clock bin
            inst_charge, _ = np.histogram(timeseries[:,0],
                                          weights = timeseries[:,2],
                                          bins = arrival_time_bin_edges)

            hold_length = self.readout_config['pixels']['integration_length']            

            # search along the bins until no more threshold crossings
            no_hits = False
            while not no_hits:
                window_charge = np.convolve(inst_charge,
                                            np.ones(hold_length))
                window_charge = window_charge[hold_length-1:]

                threshold = self.readout_config['pixels']['noise']*self.readout_config['pixels']['threshold_sigma']
                threshold_crossing_mask = window_charge > threshold
                threshold_crossing_mask *= inst_charge > 0
                
                if np.any(threshold_crossing_mask):
                    hit_index = np.where(threshold_crossing_mask)[0][0]

                    threshold_crossing_t = arrival_time_bin_edges[:-1][hit_index]
                    threshold_crossing_z = threshold_crossing_t*1.6e5 # is there a better way to do this?
                    threshold_crossing_charge = window_charge[hit_index]

                    if not nonoise:
                        # add quiescent noise
                        # threshold_crossing_charge += np.random.normal(scale = self.readout_config['pixels']['noise'])
                        threshold_crossing_charge += np.random.poisson(lam = self.readout_config['pixels']['noise'],
                                                                       size = threshold_crossing_charge)

                    inst_charge[:hit_index+hold_length] = 0

                    hits.append(PixelSample(pixel_center,
                                            threshold_crossing_t,
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
 
    def simulate(self, track, **kwargs):
        self.drift(track, **kwargs)
        self.readout(track, **kwargs)
        
    def drift(self, sampled_track, **kwargs):
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

        # TODO: implement better drift model (maybe a functional response model)
        drift_time = drift_distance/self.physics_params['charge_drift']['drift_speed'] # s
        # D accordint to https://lar.bnl.gov/properties/trans.html#diffusion-l
        # sigma = sqrt(2*D*t)

        # use the nominal drift time to calculate diffusion
        # then, add the appropriate arrival time dispersion later
        sigma_transverse = np.sqrt(2*self.physics_params['charge_drift']['diffusion_transverse']*drift_time)
        sigma_longitudinal = np.sqrt(2*self.physics_params['charge_drift']['diffusion_longitudinal']*drift_time)
        diffusion_sigma = np.array([sigma_transverse,
                                    sigma_transverse,
                                    sigma_longitudinal,
                                    ]).T
        
        drifted_positions = np.random.normal(loc = region_position,
                                             scale = diffusion_sigma*np.ones_like(region_position))

        # charge is diminished by attenuation
        drifted_charges = region_charges*np.exp(-drift_time/self.physics_params['charge_drift']['electron_lifetime'])

        # add dispersion to the arrival of charge due to longitudinal diffusion
        time_dispersion = (drifted_positions[:, 2] - region_position[:, 2])/self.physics_params['charge_drift']['drift_speed'] 
        
        if np.any(sampled_track.raw_track['times']):
            arrival_times = drift_time + sampled_track.raw_track['times'][region_mask] + time_dispersion
        else:
            arrival_times = drift_time + time_dispersion
            
        # might also include a sub-sampling step?
        # in case initial sampling is not fine enough

        sampled_track.drifted_track = {'position': drifted_positions,
                                       'charge': drifted_charges,
                                       'times': arrival_times}
        return 

    def readout(self, drifted_track, **kwargs):
        # apply the readout simulation to the drifted track
        # this is defined by the GAMPixModel object

        self.readout_model.electronics_simulation(drifted_track, **kwargs)

        return
