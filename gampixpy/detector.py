from readout_objects import DataObject, PixelSample, CoarseGridSample

class GAMPixModel:
    def __init__(self, readout_config):
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
    def __init__(self, detector_params, physics_params, readout_model):
        self.detector_params = detector_params
        self.physics_params = physics_params
        self.readout_model = readout_model

    def recombination(self, raw_track):
        # apply a recombination model to a raw track
        # and perform point sampling
        # save this representation to the track

        return None
        
    def drift(self, sampled_track):
        # drift the charge samples from their input position
        # to the anode position as defined by detector_params
        # save the drifted positions to te track
        
        return None

    def readout(self, drifted_track):
        # apply the readout simulation to the drifted track
        # this is defined by the GAMPixModel object

        return self.readout_model(drifted_track)
