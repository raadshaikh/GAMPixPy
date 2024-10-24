from tracks import Track, Segment

class InputParser:
    def __init__(self, input_filename, sequential_sampling = True):
        self.input_filename = input_filename

        self.sampling_order = []
        self.sample_cursor = 0
        
        self.generate_sample_order(sequential_sampling)

    def generate_sample_order(sequential_sampling):
        # implement!
        # page through the file, enumerate the number of samples,
        # and generate a list of indices
        # if sequential_sampling, this should just be range(n_samples)
        
        self.sampling_order = []

    def get_sample(self, index):
        # dummy method to be replaced by subclasses
        
        return None
        
    def get_next_sample(self):
        for sample_index in self.sampling_order:
            yield self.get_sample(sample_index)

class PenelopeParser (InputParser):
    def get_penelope_sample(self, sample_index):
        # do the magic that lets you read from a penelope file

    def get_sample(self, index):
        return self.get_penelope_sample(index)

class RooTrackerParser (InputParser):
    def get_G4_sample(self, sample_index):
        # do the magic that lets you read from a Geant4 ROOT file

    def get_sample(self, index):
        return self.get_G4_sample(index)
