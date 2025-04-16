import gampixpy
from gampixpy import units, mobility

import os
import yaml
import numpy as np

class Config (dict):
    def __init__(self, config_filename):
        self.config_filename = config_filename

        self.parse_config()
        self.compute_derived_parameters()
        
    def parse_config(self):
        # parse a config (yaml) file and store values
        # internally as a dict
        with open(self.config_filename) as config_file:
            self.update(yaml.load(config_file, Loader = yaml.FullLoader).items())

        # where quantities with units are specified,
        # resolve them to the internal unit system
        self.update(self.resolve_units(self))

    def resolve_units(self, sub_dict):
        if 'value' in sub_dict and 'unit' in sub_dict:
            numerical_value = sub_dict['value']
            unit = units.unit_parser(sub_dict['unit'])
            resolved_dict = numerical_value*unit
        else:
            resolved_dict = {}
            for key, value in sub_dict.items():
                if type(value) == dict:
                    resolved_dict[key] = self.resolve_units(value)
                else:
                    resolved_dict[key] = value

        return resolved_dict

class DetectorConfig (Config):
    def compute_derived_parameters(self):
        # compute any required parameters from
        # those specified in the YAML

        return

class PhysicsConfig (Config):
    def compute_derived_parameters(self):
        mobility_model = mobility.MobilityModel(self)
        self['charge_drift'].update(mobility_model.compute_parameters())

        return

class ReadoutConfig (Config):
    def compute_derived_parameters(self):
        # compute any required parameters from
        # those specified in the YAML

        # need to handle the case where the pitch doesn't evenly divide the span of the anode
        self['n_pixels_x'] = int((self['anode']['x_upper_bound'] - self['anode']['x_lower_bound'])/self['pixels']['pitch'])
        self['n_pixels_y'] = int((self['anode']['y_upper_bound'] - self['anode']['y_lower_bound'])/self['pixels']['pitch'])

        self['n_tiles_x'] = int((self['anode']['x_upper_bound'] - self['anode']['x_lower_bound'])/self['coarse_tiles']['pitch'])
        self['n_tiles_y'] = int((self['anode']['y_upper_bound'] - self['anode']['y_lower_bound'])/self['coarse_tiles']['pitch'])

        self['tile_volume_edges'] = (np.linspace(self['anode']['x_lower_bound'],
                                                 self['anode']['x_upper_bound'],
                                                 self['n_tiles_x']+1),
                                     np.linspace(self['anode']['y_lower_bound'],
                                                 self['anode']['y_upper_bound'],
                                                 self['n_tiles_y']+1))
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
