import gampixpy
import os
import yaml

class Config (dict):
    def __init__(self, config_filename):
        self.config_filename = config_filename

        self.parse_config()

    def parse_config(self):
        # parse a config (yaml) file and store values
        # internally as a dict
        with open(self.config_filename) as config_file:
            for key, value in yaml.load(config_file, Loader = yaml.FullLoader).items():
                self[key] = value
        
        return 

default_detector_params = Config(os.path.join(gampixpy.__path__[0],
                                              'detector_config',
                                              'default.yaml'))
default_physics_params = Config(os.path.join(gampixpy.__path__[0],
                                             'physics_config',
                                             'default.yaml'))
default_readout_params = Config(os.path.join(gampixpy.__path__[0],
                                             'readout_config',
                                             'default.yaml'))
