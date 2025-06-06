# This script is mean to demonstrate processing a single input
# from a provided input file, and showing how to use the built-in
# event display methods

# from gampixpy import detector, input_parsing, plotting, config, output

import importlib.util
import sys
spec = importlib.util.spec_from_file_location("gampixpy", "../gampixpy/__init__.py")
spam = importlib.util.module_from_spec(spec)
sys.modules["gampixpy"] = spam
spec.loader.exec_module(spam)
from gampixpy import detector, input_parsing, plotting, config, output

import matplotlib.pyplot as plt
import numpy as np

import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    # Set the default device to CUDA
    torch.set_default_device(device)
    print(f"Default device set to: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device('cpu')
    print("CUDA is not available, using CPU")

import pickle
def save(filename, *args):
    # Get global dictionary
    glob = globals()
    d = {}
    for v in args:
        # Copy over desired values
        d[v] = glob[v]
    with open(filename, 'wb') as f:
        # Put them in the file 
        pickle.dump(d, f)

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

    # choose the correct input parser using the provided args
    # default value for `args.input_format` is 'edepsim'
    # so this will create an EdepSimParser object and expect hdf5 input
    input_parser = input_parsing.parser_dict[args.input_format](args.input_file)
    
    raw_tracks = []
    drifted_tracks = []
    if args.output_file:
            om = output.OutputManager(args.output_file)
    for i in range(input_parser.n_events):
    # for i in range(5,6): #dry run with just one event
        event_data = input_parser.get_sample(i)
        event_meta = input_parser.get_meta(i)
        
        # print(event_meta[0])
        
        ##i was trying to find the maximum and minimum xyz positions of all sampled points in an event
        # print(torch.min(event_data.raw_track['position'][:,0]).item(),torch.max(event_data.raw_track['position'][:,0]).item())
        # print(torch.min(event_data.raw_track['position'][:,1]).item(),torch.max(event_data.raw_track['position'][:,1]).item())
        # print(torch.min(event_data.raw_track['position'][:,2]).item(),torch.max(event_data.raw_track['position'][:,2]).item())
        # print('')
        # plt.scatter(np.arange(len(event_data.raw_track['position'][::1000,2])),np.array(event_data.raw_track['position'][::1000,2]))
        # plt.show()
        

        # call the detector sim in two steps:
        detector_model.drift(event_data) # generates drifted_track attribute
        detector_model.readout(event_data) # generates pixel_samples and coarse_tile_samples
        
        # # call the detector sim in one step:
        # detector_model.simulated(event_data)

        # inspect the simulation products
        # print (event_data.raw_track) # track after recombination and point sampling
        # print (event_data.drifted_track) # track after drifting (diffusion, attenuation)
        # for h in event_data.coarse_tiles_samples:
            # print(h.coarse_measurement_time, h.coarse_measurement_depth)
        
        raw_tracks.append(event_data.raw_track)
        drifted_tracks.append(event_data.drifted_track)

        # make the event display
        # evd = plotting.EventDisplay(event_data)

        # evd.plot_raw_track(masking='none')
        # evd.plot_drifted_track()

        # methods where the z-axis is readout time
        # evd.plot_drifted_track_timeline()
        # evd.plot_drifted_track_timeline(alpha = 0) # can also pass kwargs to plt.scatter
        # evd.plot_coarse_tile_measurement_timeline(readout_config) # plot tile hits
        # evd.plot_pixel_measurement_timeline(readout_config) # plot pixel hits

        # evd.show()

        # evd.save(args.plot_output)
        # evd.save('raw_track_{}.png'.format(i))

        # save the simulation products to an hdf5 file
        if args.output_file:
            # om = output.OutputManager(args.output_file)
            om.add_entry(event_data, event_meta, event_id=i)
    # save('1-2GeVmuons_RT', 'raw_tracks') #not working for some reason
    with open('1-2GeVmuons_RT_p5_1', 'wb') as f:
        pickle.dump({'raw_tracks':raw_tracks}, f)
    with open('1-2GeVmuons_DT_p5_1', 'wb') as f:
        pickle.dump({'drifted_tracks':drifted_tracks}, f)
        
    # event_energies = [meta['primary energy'] for meta in event_metas]
    # n_coarse_hits = [len(track.coarse_tiles_samples) for track in event_datas]
    # n_fine_hits = [len(track.pixel_samples) for track in event_datas]
    
    ## this was to plot number of hits vs energy
    # event_energies = np.array(event_energies)
    # n_coarse_hits = np.array(n_coarse_hits)
    # n_fine_hits = np.array(n_fine_hits)
    # plt.subplot(1,2,1)
    # plt.scatter(event_energies, n_coarse_hits)
    # plt.xlabel('primary muon energy (MeV)')
    # plt.ylabel('no. of coarse hits')
    # plt.subplot(1,2,2)
    # plt.scatter(event_energies, n_fine_hits)
    # plt.xlabel('primary muon energy (MeV)')
    # plt.ylabel('no. of fine hits')
    # plt.show()

    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',
                        type = str,
                        help = 'input file from which to read and simulate an event')
    parser.add_argument('-i', '--input_format',
                        type = str,
                        default = 'edepsim',
                        help = 'input file format.  Must be one of {root, edepsim, marley}')
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
