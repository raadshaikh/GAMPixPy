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
    def __init__(self, pixel_id, hit_timestamp, hit_measurement):
        self.pixel_id = pixel_id
        self.hit_timestamp = hit_timestamp
        self.hit_measurement = hit_measurement

class CoarseGridSample:
    def __init__(self, coarse_cell_id, measurement_time, coarse_cell_measurement):
        self.coarse_cell_id = coarse_cell_id
        self.coarse_measurement_time = measurement_time
        self.coarse_cell_measurement = coarse_cell_measurement
