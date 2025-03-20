import numpy as np

coarse_tile_dtype = np.dtype([("event id", "u4"),
                              ("tile x", "f4"),
                              ("tile y", "f4"),
                              ("hit z", "f4"),
                              ("hit charge", "f4"),
                              ],
                             align = True)
pixel_dtype = np.dtype([("event id", "u4"),
                        ("pixel x", "f4"),
                        ("pixel y", "f4"),
                        ("hit z", "f4"),
                        ("hit charge", "f4"),
                        ],
                       align = True)

class PixelSample:
    def __init__(self,
                 pixel_pos,
                 # pixel_ind,
                 hit_timestamp,
                 hit_depth,
                 hit_measurement):
        self.pixel_pos = pixel_pos
        # self.pixel_ind = pixel_ind
        self.hit_timestamp = hit_timestamp
        self.hit_depth = hit_depth
        self.hit_measurement = hit_measurement

class CoarseGridSample:
    def __init__(self,
                 coarse_cell_pos,
                 # coarse_cell_ind,
                 measurement_time,
                 measurement_depth,
                 coarse_cell_measurement):
        self.coarse_cell_pos = coarse_cell_pos
        # self.coarse_cell_ind = coarse_cell_ind
        self.coarse_measurement_time = measurement_time
        self.coarse_measurement_depth = measurement_depth
        self.coarse_cell_measurement = coarse_cell_measurement
