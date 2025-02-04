import numpy as np
import h5py

class Track:
    def __init__(self, sample_positions, sample_charges):
        self.raw_track = {'position': sample_positions,
                          'charge': sample_charges}

        self.drifted_track = {}

        self.pixel_samples = {}
        self.coarse_tiles_samples = {}
    def save(self, output_filename, *args, **kwargs):
        outfile = h5py.File(output_filename, 'w')

        coarse_tile_dtype = np.dtype([("tile x", "f4"),
                                      ("tile y", "f4"),
                                      ("hit z", "f4"),
                                      ("hit charge", "f4"),
                                      ],
                                     align = True)
        pixel_dtype = np.dtype([("pixel x", "f4"),
                                ("pixel y", "f4"),
                                ("hit z", "f4"),
                                ("hit charge", "f4"),
                                ],
                               align = True)
        
        coarse_tile_sample_array = np.array([[hit.coarse_cell_id[0],
                                              hit.coarse_cell_id[1],
                                              hit.coarse_measurement_time,
                                              hit.coarse_cell_measurement]
                                             for hit in self.coarse_tiles_samples],
                                            dtype = coarse_tile_dtype)
        pixel_sample_array = np.array([[hit.pixel_id[0],
                                        hit.pixel_id[1],
                                        hit.hit_timestamp,
                                        hit.hit_measurement]
                                       for hit in self.pixel_samples],
                                      dtype = pixel_dtype)

        outfile.create_dataset('coarse_hits',
                               data = coarse_tile_sample_array,
                               )
        outfile.create_dataset('pixel_hits',
                               data = pixel_sample_array,
                               )

# this seems to be relevant only for specific inputs
# let's not use this for now
# class Segment:
#     def __init__(self, start_position, end_position, energy):
#         self.start_position = start_position
#         self.end_position = end_position
#         self.ionization_energy = energy
