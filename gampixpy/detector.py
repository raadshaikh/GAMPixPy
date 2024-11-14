from config import default_detector_params, default_physics_params, default_readout_params
from readout_objects import DataObject, PixelSample, CoarseGridSample

import numpy as np

class GAMPixModel:
    def __init__(self, readout_config = default_readout_params):
        self.readout_config = readout_config

    def electronics_simulation(self, drifted_track):
        # apply the electronics response simulation to
        # a track containing a drifted charge sample

        # return a DataObject

        binned_charges = self.transverse_binning(drifted_track)
        
        pixel_samples = [PixelSample(None, None, None),
                         ]
        coarse_grid_samples = [CoarseGridSample(None, None),
                               ]
        
        return DataObject(pixel_samples, coarse_grid_samples)

    def hit_finding(self, binned_charges):

        # for each bin (fine pixel/coarse tile)
        # walk along the time distribution
        # accumulating charge
        # and look for the first point
        # when the detection threshold is crossed
        # then, form a "hit" with that amount of charge
        # plus some digitization noise

        hits = []

        return hits
    
    def transverse_binning(self, drifted_track):
        
        # bin in the transverse direction
        # that is, find the pixels/coarse tiles
        # onto which the charges will be projected

        # for these bins, return the charge
        # and time distributions

        coarse_grid_timeseries = []
        fine_grid_timeseries = []
        
        return binned_charges

class DetectorModel:
    def __init__(self,
                 detector_params = default_detector_params,
                 physics_params = default_physics_params,
                 readout_params = default_readout_params):
        self.detector_params = detector_params
        self.physics_params = physics_params
        self.readout_model = GAMPixModel(readout_params)
 
    def simulate(self, track):
        self.drift(track)
        self.readout(track)
        
    def drift(self, sampled_track):
        # drift the charge samples from their input position
        # to the anode position as defined by detector_params
        # save the drifted positions to te track

        anode_z = self.detector_params['anode']['z']

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
        drift_time = drift_distance/self.physics_params['charge_drift']['velocity']

        # TODO: better diffusion model
        # D accordint to https://lar.bnl.gov/properties/trans.html#diffusion-l
        # sigma = sqrt(2*D*t)
        diffusion_sigma = np.array([self.physics_params['spatial_resolution']['sigma_transverse'],
                                    self.physics_params['spatial_resolution']['sigma_transverse'],
                                    self.physics_params['spatial_resolution']['sigma_longitudinal'],
                                    ])
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
