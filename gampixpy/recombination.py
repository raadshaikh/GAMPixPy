import torch

class RecombinationModel:
    def __init__(self, physics_config):
        self.physics_config = physics_config
        self.init_params()

    def __call__(self, *args, **kwargs):
        return self.recombination_factor(*args, **kwargs)

class BoxRecombinationModel (RecombinationModel):
    def init_params(self):
        self.E_field = self.physics_config['charge_drift']['drift_field']
        self.LAr_density = self.physics_config['material']['density']

        self.alpha = self.physics_config['box_model']['box_alpha']
        self.beta = self.physics_config['box_model']['box_beta']
        
    def recombination_factor(self, dE, dx, dEdx, **kwargs):
        xi = self.beta*dEdx/(self.E_field*self.LAr_density)
        recomb = torch.max(torch.stack([torch.zeros_like(dE),
                                        torch.log(self.alpha + xi)/xi]),
                           dim = 0)[0]

        return recomb

class BirksRecombinationModel (RecombinationModel):
    def init_params(self):
        self.E_field = self.physics_config['charge_drift']['drift_field']
        self.LAr_density = self.physics_config['material']['density']

        self.ab = self.physics_config['birks_model']['birks_ab']
        self.kb = self.physics_config['birks_model']['birks_kb']
        
    def recombination_factor(self, dE, dx, dEdx, **kwargs):
        recomb = torch.max(torch.stack([torch.zeros_like(dE),
                                        self.ab/(1 + self.kb*dEdx/(self.E_field*self.LAr_density))]),
                           dim = 0)[0]

        return recomb
