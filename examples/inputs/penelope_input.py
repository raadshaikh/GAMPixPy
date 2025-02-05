from gampixpy import input_parsing

def main(args):
    #   Need path + file name, minus extension.
    path = '/Users/tshutt/Documents/Work/Simulations/Penelope/Tracks/LAr/electrons/E0002000'
    file = 'TrackE0002000_D20241005_T1428087126'
    full_file_name = os.path.join(path, file)

    # parser = input_parsing.PenelopeParser(full_file_name)
    parser = input_parsing.InputParser(full_file_name)
    # parser.get_penelope_sample()

    # #   Read track from disk.  Note that a default charge readout has been
    # #   defined at this point (or could have been specified)
    # track = tracks_tools.Tracks(full_file_name)

    # #   Display the raw track
    # track.display(raw=True, units='cm')

    # #   Reset the parameters to chose the readout
    # track.reset_params(charge_readout_name='GAMPixG')

    # #   Readut charge, with an optional depth applied
    # depth = 0.02
    # track.readout_charge(depth=depth)
    
    # #   This is the sum of the triggered pixel samples
    # print(f'Pixel sum signal: {track.pixel_samples["samples_triggered"].sum():5.0f} e-' )

    # #   Display pixel data
    # track.display(raw=False, pixels=True, units='cm')

    return None

if __name__ == '__main__':
    import argparse
