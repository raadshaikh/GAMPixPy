from gampixpy import detector, input_parsing

def main(args):

    edepsim_track = input_parsing.EdepSimParser(args.input_edepsim_file)

    detector_model = detector.DetectorModel(None, None, None)

    detector_model.drift(edepsim_track)
    detector_model.readout(edepsim_track)
    
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
