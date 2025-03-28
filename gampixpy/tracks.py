import numpy as np

from gampixpy.readout_objects import coarse_tile_dtype, pixel_dtype

class Track:
    def __init__(self, sample_4vec, sample_charges):
        self.raw_track = {'4vec': sample_4vec,
                          'charge': sample_charges}

        self.drifted_track = {}

        self.pixel_samples = {}
        self.coarse_tiles_samples = {}

    def to_array(self):
        coarse_tile_sample_array = np.array([(0,
                                              hit.coarse_cell_pos[0],
                                              hit.coarse_cell_pos[1],
                                              hit.coarse_measurement_time,
                                              hit.coarse_cell_measurement)
                                             for hit in self.coarse_tiles_samples],
                                            dtype = coarse_tile_dtype)
        pixel_sample_array = np.array([(0,
                                        hit.pixel_pos[0],
                                        hit.pixel_pos[1],
                                        hit.hit_timestamp,
                                        hit.hit_measurement)
                                       for hit in self.pixel_samples],
                                      dtype = pixel_dtype)

        return coarse_tile_sample_array, pixel_sample_array

# this seems to be relevant only for specific inputs
# let's not use this for now
# class Segment:
#     def __init__(self, start_position, end_position, energy):
#         self.start_position = start_position
#         self.end_position = end_position
#         self.ionization_energy = energy
