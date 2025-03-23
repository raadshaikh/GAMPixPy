import tqdm
import os
from depth_estimation import *

def main(args):
    if args.detector_config == "":
        detector_config = config.default_detector_params
    else:
        detector_config = config.DetectorConfig(args.detector_config)

    if args.physics_config == "":
        physics_config = config.default_physics_params
    else:
        physics_config = config.PhysicsConfig(args.physics_config)

    if args.readout_config == "":
        readout_config = config.default_readout_params
    else:
        readout_config = config.ReadoutConfig(args.readout_config)

    detector_model = detector.DetectorModel(detector_params = detector_config,
                                            physics_params = physics_config,
                                            readout_params = readout_config,
                                            )
    
    label_dtype = np.dtype([("vertex x", "f4"),
                            ("vertex y", "f4"),
                            ("vertex z", "f4"),
                            ("deposited charge", "f4"),
                            ])
    labels = np.array([], dtype = label_dtype)

    metric_dtype = np.dtype([("pixel mean x", "f4"),
                             ("pixel mean y", "f4"),
                             ("pixel var x", "f4"),
                             ("pixel var y", "f4"),
                             ("pixel var t", "f4"),
                             ("pixel total q", "f4"),
                             ("coarse mean x", "f4"),
                             ("coarse mean y", "f4"),
                             ("coarse var x", "f4"),
                             ("coarse var y", "f4"),
                             ("coarse var t", "f4"),
                             ("coarse total q", "f4"),
                             ])
    metrics = np.array([], dtype = metric_dtype)

    for input_filename in os.listdir(args.input_directory):
        input_file_path = os.path.join(args.input_directory, input_filename)
        f = h5py.File(input_file_path)

        for event_index in tqdm.tqdm(np.unique(f['meta']['event id'])):
            meta_mask = f['meta']['event id'] == event_index
            event_meta = f['meta'][meta_mask]
        
            coarse_hits_mask = f['coarse_hits']['event id'] == event_index
            event_coarse_hits = f['coarse_hits'][coarse_hits_mask]
        
            pixel_hits_mask = f['pixel_hits']['event id'] == event_index
            event_pixel_hits = f['pixel_hits'][pixel_hits_mask]

            try:
                event_metrics = get_simple_metrics(event_pixel_hits,
                                                   event_coarse_hits)
                these_metrics = np.array([event_metrics],
                                         dtype = metric_dtype)
                metrics = np.concatenate((metrics, these_metrics))
            
                these_labels = np.array([(event_meta['vertex x'],
                                          event_meta['vertex y'],
                                          event_meta['vertex z'],
                                          event_meta['deposited charge'],
                                          )],
                                        dtype = label_dtype)
                labels = np.concatenate((labels, these_labels))
            except ZeroDivisionError:
                continue

    outfile = h5py.File(args.output_filename, 'w')

    outfile.create_dataset('labels',
                           shape = labels.shape,
                           dtype = label_dtype,
                           maxshape = (None,))
    outfile['labels'][:] = labels

    outfile.create_dataset('metrics',
                           shape = metrics.shape,
                           dtype = metric_dtype,
                           maxshape = (None,))
    outfile['metrics'][:] = metrics

    outfile.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_directory',
                        type = str,
                        help = 'input directory to search for files from which to read a simulated event')
    # parser.add_argument('-e', '--event_index',
    #                     type = int,
    #                     default = 5,
    #                     help = 'index of the event within the input file to be simulated')
    parser.add_argument('-o', '--output_filename',
                        type = str,
                        default = "",
                        help = 'output hdf5 file to store labels and output metrics')

    parser.add_argument('-d', '--detector_config',
                        type = str,
                        default = "",
                        help = 'detector configuration yaml')
    parser.add_argument('-p', '--physics_config',
                        type = str,
                        default = "",
                        help = 'physics configuration yaml')
    parser.add_argument('-r', '--readout_config',
                        type = str,
                        default = "",
                        help = 'readout configuration yaml')

    args = parser.parse_args()

    main(args)

