class Track:
    def __init__(self, sample_positions, sample_charges):
        self.positions = sample_positions
        self.charges = sample_charges

        self.drifted_positions = []
        self.drifted_charges = []

# this seems to be relevant only for specific inputs
# let's not use this for now
# class Segment:
#     def __init__(self, start_position, end_position, energy):
#         self.start_position = start_position
#         self.end_position = end_position
#         self.ionization_energy = energy
