from config import default_detector_params, default_physics_params, default_readout_params
from readout_objects import DataObject, PixelSample, CoarseGridSample

class GAMPixModel:
    def __init__(self, readout_config = default_readout_params):
        self.readout_config = readout_config

    def electronics_simulation(self, drifted_track):
        # apply the electronics response simulation to
        # a track containing a drifted charge sample

        # return a DataObject

        pixel_samples = [PixelSample(None, None, None),
                         ]
        coarse_grid_samples = [CoarseGridSample(None, None),
                               ]
        
        return DataObject(pixel_samples, coarse_grid_samples)

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

        drift_distance = self.detector_params
        return 

    def readout(self, drifted_track):
        # apply the readout simulation to the drifted track
        # this is defined by the GAMPixModel object

        self.readout_model(drifted_track)

        return
