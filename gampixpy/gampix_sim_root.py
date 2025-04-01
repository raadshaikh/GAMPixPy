import gampixpy
from gampixpy import detector, input_parsing, plotting, config, output

import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    # Set the default device to CUDA
    torch.set_default_device(device)
    print(f"Default device set to: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device('cpu')
    print("CUDA is not available, using CPU")


def main(args):

    # load configs for physics, detector, and readout

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

    input_parser = input_parsing.RooTrackerParser(args.input_root_file)
    root_track = input_parser.get_sample(args.event_index)
    root_event_meta = input_parser.get_meta(args.event_index)

    detector_model.drift(root_track)

    detector_model.readout(root_track)

    if args.plot_output:
        evd = plotting.EventDisplay(root_track)
        evd.init_fig()
        evd.plot_drifted_track_timeline(alpha = 0)
        # evd.plot_drifted_track()
        evd.plot_coarse_tile_measurement_timeline(readout_config)
        evd.plot_pixel_measurement_timeline(readout_config)
        # print ([sample.pixel_pos for sample in  root_track.pixel_samples])
        # evd.plot_raw_track()
        evd.show()

        evd.save(args.plot_output)

    if args.output_file:
        om = output.OutputManager(args.output_file)
        om.add_entry(root_track, root_event_meta)
        # om.add_track(root_track)

    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_root_file',
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
    parser.add_argument('--plot_output',
                        type = str,
                        default = "",
                        help = 'file to save output plot')

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
