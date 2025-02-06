import gampixpy

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

        # Using the mobility model described in https://arxiv.org/abs/1508.07059
        a0 = self['charge_drift']['mobility_parameter_a0'] # cm^2/V/s
        a1 = self['charge_drift']['mobility_parameter_a1']
        a2 = self['charge_drift']['mobility_parameter_a2']
        a3 = self['charge_drift']['mobility_parameter_a3']
        a4 = self['charge_drift']['mobility_parameter_a4']
        a5 = self['charge_drift']['mobility_parameter_a5']
        T0 = self['charge_drift']['mobility_reference_T0']

        b0 = self['charge_drift']['mobility_parameter_b0'] # eV
        b1 = self['charge_drift']['mobility_parameter_b1']
        b2 = self['charge_drift']['mobility_parameter_b2']
        b3 = self['charge_drift']['mobility_parameter_b3']
        T1 = self['charge_drift']['mobility_reference_T1']

        E = self['charge_drift']['drift_field'] # kV/cm
        T = self['material']['temperature'] # K
        
        numerator = a0 + a1*E + a2*np.pow(E, 3./2) + a3*np.pow(E, 5./2)
        denominator = 1 + (a1/a0)*E + a4*np.pow(E, 2) + a5*np.pow(E, 3)
        mu = numerator*np.pow(T/T0, -3./2)/denominator # cm^2/V/s
        # mu *= 10*10 # mm^2/V/s
        # mu /= 1.e9 # mm^2/V/ns
        
        dMudE = ((a1 + 3./2*a2*np.pow(E, 1./2) + 5./2*a3*np.pow(E, 3./2))*denominator - numerator*(a1/a0 + 2*a4*E + 3*a5*np.pow(E, 2)))*np.pow(T/T0, -3./2)/np.pow(denominator, 2) # cm^3/V/kV/s
        dMudE /= 1000.
        # dMudE *= 10*10*10 # mm^3/V^2/s
        # dMudE /= 1.e9 # mm^3/V^2/ns

        e = (b0 + b1*E + b2*np.pow(E, 2))*(T/T1)/(1 + (b1/b0)*E + b3*np.pow(E, 2)) # eV
        e /= 1.e6 # MeV

        R = 1 + E/mu*dMudE*1000 # diffusion ratio
        
        self['charge_drift']['electron_mobility'] = mu # cm^2/V/s
        self['charge_drift']['drift_energy'] = e # MeV
        self['charge_drift']['diffusion_longitudinal'] = mu*e*1.e6 # cm^2/s
        self['charge_drift']['diffusion_ratio'] = R
        self['charge_drift']['diffusion_transverse'] = mu*e/R*1.e6 # cm^2/s
        self['charge_drift']['drift_speed'] = mu*E*1.e3 # cm/s

        return

class ReadoutConfig (Config):
    def compute_derived_parameters(self):
        # compute any required parameters from
        # those specified in the YAML

        # need to handle the case where the pitch doesn't evenly divide the span of the anode
        self['n_pixels_x'] = int((self['anode']['x_range'][1] - self['anode']['x_range'][0])/self['pixels']['pitch'])
        self['n_pixels_y'] = int((self['anode']['y_range'][1] - self['anode']['y_range'][0])/self['pixels']['pitch'])

        self['n_tiles_x'] = int((self['anode']['x_range'][1] - self['anode']['x_range'][0])/self['coarse_tiles']['pitch'])
        self['n_tiles_y'] = int((self['anode']['y_range'][1] - self['anode']['y_range'][0])/self['coarse_tiles']['pitch'])

        self['tile_volume_edges'] = (np.linspace(self['anode']['x_range'][0],
                                                 self['anode']['x_range'][1],
                                                 self['n_tiles_x']+1),
                                     np.linspace(self['anode']['y_range'][0],
                                                 self['anode']['y_range'][1],
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
