import torch

class MobilityModel:
    def __init__(self, physics_config):
        self.physics_config = physics_config

    def mobility_parameters(self):
        # Using the mobility model described in https://arxiv.org/abs/1508.07059
        drift_params = self.physics_config['charge_drift']
        
        a0 = drift_params['mobility_parameter_a0'] # cm^2/V/s
        a1 = drift_params['mobility_parameter_a1']
        a2 = drift_params['mobility_parameter_a2']
        a3 = drift_params['mobility_parameter_a3']
        a4 = drift_params['mobility_parameter_a4']
        a5 = drift_params['mobility_parameter_a5']
        T0 = drift_params['mobility_reference_T0']

        b0 = drift_params['mobility_parameter_b0']
        b1 = drift_params['mobility_parameter_b1']
        b2 = drift_params['mobility_parameter_b2']
        b3 = drift_params['mobility_parameter_b3']
        T1 = drift_params['mobility_reference_T1']

        E = drift_params['drift_field']
        T = self.physics_config['material']['temperature']

        numerator = a0 + a1*E + a2*np.power(E, 3./2) + a3*np.power(E, 5./2)
        denominator = 1 + (a1/a0)*E + a4*np.power(E, 2) + a5*np.power(E, 3)
        mu = numerator*np.power(T/T0, -3./2)/denominator # cm^2/V/s
        # mu *= 10*10 # mm^2/V/s
        # mu /= 1.e9 # mm^2/V/ns
        
        dMudE = ((a1 + 3./2*a2*np.power(E, 1./2) + 5./2*a3*np.power(E, 3./2))*denominator - numerator*(a1/a0 + 2*a4*E + 3*a5*np.power(E, 2)))*np.power(T/T0, -3./2)/np.power(denominator, 2) # cm^3/V/kV/s
        dMudE /= 1000.
        # dMudE *= 10*10*10 # mm^3/V^2/s
        # dMudE /= 1.e9 # mm^3/V^2/ns

        e = (b0 + b1*E + b2*np.power(E, 2))*(T/T1)/(1 + (b1/b0)*E + b3*np.power(E, 2)) # eV
        e /= 1.e6 # MeV

        R = 1 + E/mu*dMudE*1000 # diffusion ratio

        computed_params = {}
        computed_params['electron_mobility'] = mu # cm^2/V/s
        computed_params['drift_energy'] = e # MeV
        computed_params['diffusion_longitudinal'] = mu*e*1.e6 # cm^2/s
        computed_params['diffusion_ratio'] = R
        computed_params['diffusion_transverse'] = mu*e/R*1.e6 # cm^2/s
        computed_params['drift_speed'] = mu*E*1.e3 # cm/s

        return computed_params
