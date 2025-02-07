import tqdm

import gampixpy
from gampixpy import detector, input_parsing, plotting, config, output

def main(args):

    # load configs for physics, detector, and readout
    import os
    gampixD_readout_config = config.ReadoutConfig(os.path.join(gampixpy.__path__[0],
                                                               'readout_config',
                                                               'GAMPixD.yaml'))
    detector_model = detector.DetectorModel(readout_params = gampixD_readout_config,
                                            )

    if args.output_file:
        output_manager = output.OutputManager(args.output_file)

    input_parser = input_parsing.EdepSimParser(args.input_edepsim_file)
    for event_index, edepsim_track in tqdm.tqdm(input_parser):
        detector_model.simulate(edepsim_track)
        print ("found", len(edepsim_track.coarse_tiles_samples), "coarse tile hits")
        print ("found", len(edepsim_track.pixel_samples), "pixel hits")
        
        if args.output_file:
            output_manager.add_track(edepsim_track)

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
