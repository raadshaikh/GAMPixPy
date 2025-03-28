import h5py
import numpy as np

from gampixpy.readout_objects import coarse_tile_dtype, pixel_dtype
from gampixpy.input_parsing import meta_dtype

class OutputManager:
    def __init__(self, output_filename):
        self.output_filename = output_filename

        self.outfile = h5py.File(output_filename, 'w')

        self.outfile.create_dataset('coarse_hits',
                                    shape = (0,),
                                    dtype = coarse_tile_dtype,
                                    maxshape = (None,))
        self.outfile.create_dataset('pixel_hits',
                                    shape = (0,),
                                    dtype = pixel_dtype,
                                    maxshape = (None,))
        self.outfile.create_dataset('meta',
                                    shape = (0,),
                                    dtype = meta_dtype,
                                    maxshape = (None,))
        self.n_tracks = 0 # track how many tracks have been written so far

    def add_entry(self, track, meta, event_id = None):
        self.add_track(track, event_id)
        self.add_meta(meta, event_id)

        self.n_tracks += 1
        
    def add_track(self, track, event_id = None):
        coarse_tile_sample_array, pixel_sample_array = track.to_array()

        if event_id:
            coarse_tile_sample_array[:]['event id'] = event_id
            pixel_sample_array[:]['event id'] = event_id
        else:
            coarse_tile_sample_array[:]['event id'] = self.n_tracks
            pixel_sample_array[:]['event id'] = self.n_tracks
        
        n_coarse_hits = coarse_tile_sample_array.shape[0]
        n_coarse_hits_prev = self.outfile['coarse_hits'].shape[0]

        n_pixel_hits = pixel_sample_array.shape[0]
        n_pixel_hits_prev = self.outfile['pixel_hits'].shape[0]

        self.outfile['coarse_hits'].resize((n_coarse_hits+n_coarse_hits_prev,))
        self.outfile['coarse_hits'][n_coarse_hits_prev:] = coarse_tile_sample_array

        self.outfile['pixel_hits'].resize((n_pixel_hits+n_pixel_hits_prev,))
        self.outfile['pixel_hits'][n_pixel_hits_prev:] = pixel_sample_array

    def add_meta(self, meta, event_id):
        if event_id:
            meta[:]['event id'] = event_id
        else:
            meta[:]['event id'] = self.n_tracks
        
        self.outfile['meta'].resize((self.n_tracks + 1,))
        self.outfile['meta'][self.n_tracks:] = meta
