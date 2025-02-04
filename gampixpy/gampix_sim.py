import gampixpy
from gampixpy import detector, input_parsing, plotting, config

def main(args):

    # load configs for physics, detector, and readout

    # detector_model = detector.DetectorModel() # default configs
    import os
    gampixD_readout_config = config.ReadoutConfig(os.path.join(gampixpy.__path__[0],
                                                               'readout_config',
                                                               'GAMPixD.yaml'))
    detector_model = detector.DetectorModel(# detector_params = DetectorConfig('gampixpy/detector_config,
                                            # physics_params = ,
                                            readout_params = gampixD_readout_config,
                                            )

    input_parser = input_parsing.EdepSimParser(args.input_edepsim_file)
    edepsim_track = input_parser.get_sample(args.event_index)
    evd = plotting.EventDisplay(edepsim_track)
    # print (edepsim_track.raw_track)


    detector_model.drift(edepsim_track)
    # print (edepsim_track.drifted_track)

    # evd.init_fig()
    # evd.plot_drifted_track()

    detector_model.readout(edepsim_track)
    # print (edepsim_track.pixel_samples)
    # print (edepsim_track.coarse_tiles_samples)
    # print (edepsim_track.pixel_samples)
    # evd.plot_coarse_tile_measurement(config.default_readout_params)
    evd.plot_coarse_tile_measurement(gampixD_readout_config)
    # evd.plot_pixel_measurement(config.default_readout_params)
    # evd.plot_raw_track()
    evd.plot_drifted_track()
    evd.show()

    if args.output_file:
        edepsim_track.save(args.output_file)
        
    # # save the timing and hit magnitude distributions (optionally)
    # # for testing purposes
    # import numpy as np
    # import os
    # coarse_charges = np.array([sample.coarse_cell_measurement
    #                            for sample in edepsim_track.coarse_tiles_samples])
    # coarse_times = np.array([sample.coarse_measurement_time
    #                          for sample in edepsim_track.coarse_tiles_samples])

    # pixel_charges = np.array([sample.hit_measurement
    #                           for sample in edepsim_track.pixel_samples])
    # pixel_times = np.array([sample.hit_timestamp
    #                         for sample in edepsim_track.pixel_samples])
    # output_prefix = './'
    # np.save(os.path.join(output_prefix,
    #                      'pixel_samples.npy'),
    #         pixel_charges)
    # np.save(os.path.join(output_prefix,
    #                       'tile_samples.npy'),
    #         coarse_charges)
    # np.save(os.path.join(output_prefix,
    #                      'pixel_timing.npy'),
    #         pixel_times)
    # np.save(os.path.join(output_prefix,
    #                       'tile_timing.npy'),
    #         coarse_times)


    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_edepsim_file',
                        type = str,
                        help = 'input file from which to read and simulate an event')
    parser.add_argument('-e', '--event_index',
                        type = int,
                        default = 5,
                        help = 'index of the event within the input file to be simulated')
    parser.add_argument('-o', '--output_file',
                        type = str,
                        default = "",
                        help = 'output hdf5 file to store coarse tile and pixel measurements')

    args = parser.parse_args()

    main(args)
