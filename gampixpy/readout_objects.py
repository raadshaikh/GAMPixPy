class DataObject:
    def __init__(self, pixel_samples, coarse_grid_samples):
        self.pixel_samples = pixel_samples
        self.coarse_grid_samples = coarse_grid_samples

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
