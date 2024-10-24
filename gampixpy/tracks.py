class Track:
    def __init__(self, segment_list):
        self.depositions = segment_list

        self.charge_unit = 0
        self.charge_samples = []
        self.drifted_samples = []

class Segment:
    def __init__(self, start_position, end_position, energy):
        self.start_position = start_position
        self.end_position = end_position
        self.ionization_energy = energy
