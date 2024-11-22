from gampixpy import detector, input_parsing, plotting, config

def main(args):

    input_parser = input_parsing.EdepSimParser(args.input_edepsim_file)
    edepsim_track = input_parser.get_edepsim_event(args.event_index)
    evd = plotting.EventDisplay(edepsim_track)
    # print (edepsim_track.raw_track)

    # evd.plot_raw_track()

    detector_model = detector.DetectorModel()

    detector_model.drift(edepsim_track)
    # print (edepsim_track.drifted_track)

    # evd.init_fig()
    # evd.plot_drifted_track()

    detector_model.readout(edepsim_track)
    # print (edepsim_track.pixel_samples)
    # print (edepsim_track.coarse_tiles_samples)
    print (edepsim_track.pixel_samples)
    evd.plot_coarse_tile_measurement(config.default_readout_params)
    evd.plot_pixel_measurement(config.default_readout_params)
    # evd.plot_raw_track()
    evd.plot_drifted_track()
    evd.show()
    
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

    args = parser.parse_args()

    main(args)
