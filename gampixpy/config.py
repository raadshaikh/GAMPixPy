import gampixpy
import os
import yaml

class Config (dict):
    def __init__(self, config_filename):
        self.config_filename = config_filename

        self.parse_config()
        self.compute_derived_parameters()
        
    def parse_config(self):
        # parse a config (yaml) file and store values
        # internally as a dict
        with open(self.config_filename) as config_file:
            for key, value in yaml.load(config_file, Loader = yaml.FullLoader).items():
                self[key] = value
        
        return

class DetectorConfig (Config):
    def compute_derived_parameters(self):
        # compute any required parameters from
        # those specified in the YAML

        return

class PhysicsConfig (Config):
    def compute_derived_parameters(self):
        # compute drift velocity, diffusion model

        return

class ReadoutConfig (Config):
    def compute_derived_parameters(self):
        # compute any required parameters from
        # those specified in the YAML

        return

default_detector_params = DetectorConfig(os.path.join(gampixpy.__path__[0],
                                                      'detector_config',
                                                      'default.yaml'))
default_physics_params = PhysicsConfig(os.path.join(gampixpy.__path__[0],
                                                    'physics_config',
                                                    'default.yaml'))
default_readout_params = ReadoutConfig(os.path.join(gampixpy.__path__[0],
                                                    'readout_config',
                                                    'default.yaml'))
