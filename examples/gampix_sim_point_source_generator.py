from gampixpy import detector, generator, input_parsing, plotting, config, output

import tqdm
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

    x_range = args.x_range.split(',')
    x_range = [float(x_range[0]), float(x_range[1])]

    y_range = args.y_range.split(',')
    y_range = [float(y_range[0]), float(y_range[1])]

    z_range = args.z_range.split(',')
    z_range = [float(z_range[0]), float(z_range[1])]

    q_range = args.q_range.split(',')
    q_range = [float(q_range[0]), float(q_range[1])]

    ps_generator = generator.PointSource(x_range = x_range,
                                         y_range = y_range,
                                         z_range = z_range,
                                         q_range = q_range,
                                         )

    if args.output_file:
        om = output.OutputManager(args.output_file)

    for i in tqdm.tqdm(range(args.n_samples)):
        cloud_track = ps_generator.get_sample()
        cloud_meta = ps_generator.get_meta()

        detector_model.simulate(cloud_track, verbose = False)

        if args.output_file:
            om.add_entry(cloud_track, cloud_meta)

    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_file',
                        type = str,
                        default = "",
                        help = 'output hdf5 file to store coarse tile and pixel measurements')
    parser.add_argument('-n', '--n_samples',
                        type = int,
                        default = 1000,
                        help = 'number of point sources per output file')

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

    parser.add_argument('-x', '--x_range',
                        type = str,
                        default = "-10,10",
                        help = 'min,max x values over which to generate point sources (e.g. -2,4)')
    parser.add_argument('-y', '--y_range',
                        type = str,
                        default = "-10,10",
                        help = 'min,max y values over which to generate point sources (e.g. -2,4)')
    parser.add_argument('-z', '--z_range',
                        type = str,
                        default = "10,100",
                        help = 'min,max z values over which to generate point sources (e.g. -2,4)')
    parser.add_argument('-q', '--q_range',
                        type = str,
                        default = "100,100000",
                        help = 'min,max q values over which to generate point sources (e.g. -2,4)')
    

    args = parser.parse_args()

    main(args)
