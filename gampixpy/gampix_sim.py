import gampixpy
from gampixpy import detector, input_parsing, plotting, config, output

def main(args):

    # load configs for physics, detector, and readout

    # detector_model = detector.DetectorModel() # default configs
    import os
    # gampixD_readout_config = config.ReadoutConfig(os.path.join(gampixpy.__path__[0],
    #                                                            'readout_config',
    #                                                            # 'GAMPixD.yaml'))
    #                                                            'chunky_example.yaml'))
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

    input_parser = input_parsing.EdepSimParser(args.input_edepsim_file)
    edepsim_track = input_parser.get_sample(args.event_index)
    evd = plotting.EventDisplay(edepsim_track)
    # print (edepsim_track.raw_track)

    detector_model.drift(edepsim_track)
    # print (edepsim_track.drifted_track)

    # evd.init_fig()
    # evd.plot_drifted_track()

    detector_model.readout(edepsim_track)
    evd.plot_drifted_track()
    # evd.plot_coarse_tile_measurement(gampixD_readout_config)
    evd.plot_pixel_measurement(readout_config)
    # evd.plot_raw_track()
    evd.show()

    if args.output_file:
        om = output.OutputManager(args.output_file)
        om.add_track(edepsim_track)

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
