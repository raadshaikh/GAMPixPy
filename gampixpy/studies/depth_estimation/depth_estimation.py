import h5py
import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from gampixpy import config, generator, detector

def plot_boxes(ax, x_bounds, y_bounds, z_bounds, **kwargs):
    bottom_face = np.array([[x_bounds[0], x_bounds[0], x_bounds[1], x_bounds[1], x_bounds[0]],
                            [y_bounds[0], y_bounds[1], y_bounds[1], y_bounds[0], y_bounds[0]],
                            [z_bounds[0], z_bounds[0], z_bounds[0], z_bounds[0], z_bounds[0]]]).T
    top_face = np.array([[x_bounds[0], x_bounds[0], x_bounds[1], x_bounds[1], x_bounds[0]],
                         [y_bounds[0], y_bounds[1], y_bounds[1], y_bounds[0], y_bounds[0]],
                         [z_bounds[1], z_bounds[1], z_bounds[1], z_bounds[1], z_bounds[1]]]).T
    left_face = np.array([[x_bounds[0], x_bounds[0], x_bounds[0], x_bounds[0], x_bounds[0]],
                          [y_bounds[0], y_bounds[1], y_bounds[1], y_bounds[0], y_bounds[0]],
                          [z_bounds[0], z_bounds[0], z_bounds[1], z_bounds[1], z_bounds[0]]]).T
    right_face = np.array([[x_bounds[1], x_bounds[1], x_bounds[1], x_bounds[1], x_bounds[1]],
                           [y_bounds[0], y_bounds[1], y_bounds[1], y_bounds[0], y_bounds[0]],
                           [z_bounds[0], z_bounds[0], z_bounds[1], z_bounds[1], z_bounds[0]]]).T
    back_face = np.array([[x_bounds[0], x_bounds[1], x_bounds[1], x_bounds[0], x_bounds[0]],
                          [y_bounds[0], y_bounds[0], y_bounds[0], y_bounds[0], y_bounds[0]],
                          [z_bounds[0], z_bounds[0], z_bounds[1], z_bounds[1], z_bounds[0]]]).T
    front_face = np.array([[x_bounds[0], x_bounds[1], x_bounds[1], x_bounds[0], x_bounds[0]],
                           [y_bounds[1], y_bounds[1], y_bounds[1], y_bounds[1], y_bounds[1]],
                           [z_bounds[0], z_bounds[0], z_bounds[1], z_bounds[1], z_bounds[0]]]).T
    faces = [bottom_face,
             top_face,
             left_face,
             right_face,
             back_face,
             front_face,
             ]
    ax.add_collection3d(Poly3DCollection(faces, **kwargs))

def plot_coarse_hits(ax, cell_center_x_array, cell_center_y_array, cell_trigger_time_array, readout_config):
    for cell_center_x, cell_center_y, cell_trigger_time in zip(cell_center_x_array, cell_center_y_array, cell_trigger_time_array):
        x_bounds = [cell_center_x - 0.5*readout_config['coarse_tiles']['pitch'],
                    cell_center_x + 0.5*readout_config['coarse_tiles']['pitch']]
        y_bounds = [cell_center_y - 0.5*readout_config['coarse_tiles']['pitch'],
                    cell_center_y + 0.5*readout_config['coarse_tiles']['pitch']]
        z_bounds = [cell_trigger_time,
                    cell_trigger_time + readout_config['coarse_tiles']['clock_interval']*readout_config['coarse_tiles']['integration_length']]
        
        plot_boxes(ax,
                   x_bounds,
                   y_bounds,
                   z_bounds,
                   facecolors = 'yellow',
                   linewidths=1,
                   edgecolors = 'k',
                   alpha = 0.25)

def plot_pixels(ax, cell_center_x_array, cell_center_y_array, cell_trigger_time_array, readout_config):
    for cell_center_x, cell_center_y, cell_trigger_time in zip(cell_center_x_array, cell_center_y_array, cell_trigger_time_array):
        x_bounds = [cell_center_x - 0.5*readout_config['pixels']['pitch'],
                    cell_center_x + 0.5*readout_config['pixels']['pitch']]
        y_bounds = [cell_center_y - 0.5*readout_config['pixels']['pitch'],
                    cell_center_y + 0.5*readout_config['pixels']['pitch']]
        z_bounds = [cell_trigger_time,
                    cell_trigger_time + readout_config['pixels']['clock_interval']*readout_config['pixels']['integration_length']]
        
        plot_boxes(ax,
                   x_bounds,
                   y_bounds,
                   z_bounds,
                   facecolors = 'cyan',
                   linewidths=1,
                   edgecolors = 'k',
                   alpha = 0.25)

def charge_hypothesis(detector_model, source_x, source_y, source_z, source_q):
    ps_generator = generator.PointSource(x_range = [source_x, source_x],
                                         y_range = [source_y, source_y],
                                         z_range = [source_z, source_z],
                                         q_range = [source_q, source_q],
                                         )

    cloud_track = ps_generator.get_sample()
    detector_model.simulate(cloud_track, verbose = False, nonoise = True)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection = '3d')
    tile_array, pixel_array = cloud_track.to_array()
    # plot_pixels(ax,
    #             pixel_array["pixel x"],
    #             pixel_array["pixel y"],
    #             pixel_array["hit z"],
    #             detector_model.readout_params
    #             )
    # ax.scatter(source_x,
    #            source_y,
    #            source_z,
    #            marker = '*',
    #            color = 'red',
    #            )

    return tile_array, pixel_array

def hypothesis_likelihood(observed_coarse_hits,
                          observed_pixel_hits,
                          hyp_coarse_hits,
                          hyp_pixel_hits,
                          coarse_noise,
                          pixel_noise):
    
    observed_coarse_hit_hash = [hash((this_hit['tile x'], this_hit['tile y']))
                                for this_hit in observed_coarse_hits]
    observed_pixel_hit_hash = [hash((this_hit['pixel x'], this_hit['pixel y']))
                               for this_hit in observed_pixel_hits]
    hyp_coarse_hit_hash = [hash((this_hit['tile x'], this_hit['tile y']))
                           for this_hit in hyp_coarse_hits]
    hyp_pixel_hit_hash = [hash((this_hit['pixel x'], this_hit['pixel y']))
                          for this_hit in hyp_pixel_hits]

    observation_LLH = 0
    
    for this_coarse_hit_hash in np.unique(observed_coarse_hit_hash + hyp_coarse_hit_hash):
        if this_coarse_hit_hash in observed_coarse_hit_hash:
            observed_hits_mask = observed_coarse_hit_hash == this_coarse_hit_hash
            observed_hits_here = observed_coarse_hits[observed_hits_mask]
            observed_charge_here = sum(observed_hits_here['hit charge'])
            if this_coarse_hit_hash in hyp_coarse_hit_hash:
                # there are hits on this pixel in both
                hyp_hits_mask = hyp_coarse_hit_hash == this_coarse_hit_hash
                hyp_hits_here = hyp_coarse_hits[hyp_hits_mask]
                hyp_charge_here = sum(hyp_hits_here['hit charge'])
            else:
                # there are hits on this pixel in the observation
                # but not the hypothesis
                hyp_charge_here = 0
        else:
            # there are no hits on this pixel in the observation
            # but there are hits in the hypothesis
            hyp_hits_mask = hyp_coarse_hit_hash == this_coarse_hit_hash
            hyp_hits_here = hyp_coarse_hits[hyp_hits_mask]
            hyp_charge_here = sum(hyp_hits_here['hit charge'])
            observed_charge_here = 0

        # observation_likelihood *= st.poisson(hyp_charge_here).pmf(observed_charge_here)
        # observation_likelihood *= st.poisson(hyp_charge_here).pmf(observed_charge_here)
        observation_LLH += -(hyp_charge_here - observed_charge_here)**2/(2*coarse_noise**2) - np.log(np.sqrt(2*np.pi*coarse_noise**2))

    for this_pixel_hit_hash in np.unique(observed_pixel_hit_hash + hyp_pixel_hit_hash):
        if this_pixel_hit_hash in observed_pixel_hit_hash:
            observed_hits_mask = observed_pixel_hit_hash == this_pixel_hit_hash
            observed_hits_here = observed_pixel_hits[observed_hits_mask]
            observed_charge_here = sum(observed_hits_here['hit charge'])
            if this_pixel_hit_hash in hyp_pixel_hit_hash:
                # there are hits on this pixel in both
                hyp_hits_mask = hyp_pixel_hit_hash == this_pixel_hit_hash
                hyp_hits_here = hyp_pixel_hits[hyp_hits_mask]
                hyp_charge_here = sum(hyp_hits_here['hit charge'])
            else:
                # there are hits on this pixel in the observation
                # but not the hypothesis
                hyp_charge_here = 0
        else:
            # there are no hits on this pixel in the observation
            # but there are hits in the hypothesis
            hyp_hits_mask = hyp_pixel_hit_hash == this_pixel_hit_hash
            hyp_hits_here = hyp_pixel_hits[hyp_hits_mask]
            hyp_charge_here = sum(hyp_hits_here['hit charge'])
            observed_charge_here = 0
        
        # observation_likelihood *= st.poisson(hyp_charge_here).pmf(observed_charge_here)
        observation_LLH += -(hyp_charge_here - observed_charge_here)**2/(2*coarse_noise**2) - np.log(np.sqrt(2*np.pi*coarse_noise**2))
        # observation_LLH += (hyp_charge_here - observed_charge_here)**2/pixel_noise**2

    return -observation_LLH

def get_simple_metrics(event_pixel_hits,
                       event_coarse_hits):
    pixel_mean_x = sum(event_pixel_hits['pixel x']*event_pixel_hits['hit charge'])/sum(event_pixel_hits['hit charge'])
    pixel_mean_y = sum(event_pixel_hits['pixel y']*event_pixel_hits['hit charge'])/sum(event_pixel_hits['hit charge'])
    pixel_mean_t = sum(event_pixel_hits['hit z']*event_pixel_hits['hit charge'])/sum(event_pixel_hits['hit charge'])

    pixel_var_x = sum(event_pixel_hits['pixel x']**2*event_pixel_hits['hit charge'])/sum(event_pixel_hits['hit charge']) - pixel_mean_x**2
    pixel_var_y = sum(event_pixel_hits['pixel y']**2*event_pixel_hits['hit charge'])/sum(event_pixel_hits['hit charge']) - pixel_mean_y**2
    pixel_var_t = sum(event_pixel_hits['hit z']**2*event_pixel_hits['hit charge'])/sum(event_pixel_hits['hit charge']) - pixel_mean_t**2

    pixel_total_q = sum(event_pixel_hits['hit charge'])

    coarse_mean_x = sum(event_coarse_hits['tile x']*event_coarse_hits['hit charge'])/sum(event_coarse_hits['hit charge'])
    coarse_mean_y = sum(event_coarse_hits['tile x']*event_coarse_hits['hit charge'])/sum(event_coarse_hits['hit charge'])
    coarse_mean_t = sum(event_coarse_hits['hit z']*event_coarse_hits['hit charge'])/sum(event_coarse_hits['hit charge'])
    coarse_var_x = sum(event_coarse_hits['tile x']**2*event_coarse_hits['hit charge'])/sum(event_coarse_hits['hit charge']) - coarse_mean_x**2
    coarse_var_y = sum(event_coarse_hits['tile y']**2*event_coarse_hits['hit charge'])/sum(event_coarse_hits['hit charge']) - coarse_mean_y**2
    coarse_var_t = sum(event_coarse_hits['hit z']**2*event_coarse_hits['hit charge'])/sum(event_coarse_hits['hit charge']) - coarse_mean_y**2
    coarse_total_q = sum(event_coarse_hits['hit charge'])

    metrics = (pixel_mean_x,
               pixel_mean_y,
               pixel_var_x,
               pixel_var_y,
               pixel_var_t,
               pixel_total_q,
               coarse_mean_x,
               coarse_mean_y,
               coarse_var_x,
               coarse_var_y,
               coarse_var_t,
               coarse_total_q)
               
    return metrics

def simple_point_source_position_estimate(event_pixel_hits,
                                          event_coarse_hits):

    metrics = get_simple_metrics(event_pixel_hits,
                                 event_coarse_hits)

    mean_x = metrics[0]
    mean_y = metrics[1]
    var_x = metrics[2]
    var_y = metrics[3]
    total_q = metrics[11]
    
    estimate_x = mean_x
    estimate_y = mean_y
    if len(event_pixel_hits) > 1:
        estimate_z = 2500*(var_x + var_y)
    else:
        estimate_z = 100
    estimate_q = total_q

    estimate = (estimate_x,
                estimate_y,
                estimate_z,
                estimate_q,
                )

    return estimate
    
def point_source_position_estimate(event_pixel_hits,
                                   event_coarse_hits,
                                   detector_model):

    from scipy.optimize import fmin_l_bfgs_b as opt
    # from scipy.optimize import fmin as opt
    # from scipy.optimize import fmin_bfgs as opt

    first_guess = simple_point_source_position_estimate(event_pixel_hits,
                                                        event_coarse_hits)

    def depth_source_nLLH(depth_hyp):
        position_hyp = (first_guess[0],
                        first_guess[1],
                        depth_hyp[0],
                        depth_hyp[1])
        
        hyp_coarse_hits, hyp_pixel_hits = charge_hypothesis(detector_model, *position_hyp)
        nLLH = hypothesis_likelihood(event_coarse_hits,
                                     event_pixel_hits,
                                     hyp_coarse_hits,
                                     hyp_pixel_hits,
                                     detector_model.readout_params['coarse_tiles']['noise'],
                                     detector_model.readout_params['pixels']['noise'],
                                     )
        return nLLH

    def point_source_nLLH(position_hyp):
        hyp_coarse_hits, hyp_pixel_hits = charge_hypothesis(detector_model, *position_hyp)
        nLLH = hypothesis_likelihood(event_coarse_hits,
                                     event_pixel_hits,
                                     hyp_coarse_hits,
                                     hyp_pixel_hits,
                                     detector_model.readout_params['coarse_tiles']['noise'],
                                     detector_model.readout_params['pixels']['noise'],
                                     )
        return nLLH
    
    print (first_guess, point_source_nLLH(first_guess))

    for z in np.linspace(first_guess[2], first_guess[2]+50, 101):
        print (z, depth_source_nLLH((z, first_guess[3])))
        
    # opt_result = opt(depth_source_nLLH,
    #                  (first_guess[2], first_guess[3]),
    #                  approx_grad = True,
    #                  epsilon = 1.e2,
    #                  )

    # print (opt_result)
    # depth_optimized_guess = (first_guess[0],
    #                          first_guess[1],
    #                          # opt_result[0][0],
    #                          opt_result[0],
    #                          first_guess[3],
    #                          )

    # opt_result = opt(point_source_nLLH,
    #                  # depth_optimized_guess,
    #                  first_guess,
    #                  approx_grad = True,
    #                  # epsilon = 1.e1,
    #                  )

    # print (opt_result)

    # best_guess = opt_result[0]
    # best_guess = opt_result[0]
    # true_args = (-1.265,
    #              -9.558,
    #              171.6978,
    #              4971.5)

    # print ("best LLH", point_source_nLLH(true_args))
    
    # hyp_coarse_hits, hyp_pixel_hits = charge_hypothesis(detector_model, *first_guess)
    # nLLH = hypothesis_likelihood(event_coarse_hits,
    #                             event_pixel_hits,
    #                             hyp_coarse_hits,
    #                             hyp_pixel_hits,
    #                             detector_model.readout_params['coarse_tiles']['noise'],
    #                             detector_model.readout_params['pixels']['noise'],
    #                             )
    # print ("first guess position", first_guess[2])
    # print ("first guess nLLH", nLLH)

    # return best_guess
    return first_guess
        
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
    
    f = h5py.File('gampixD_point_source_100-5000.h5')

    print (f.keys())
    print (f['meta'])
    print (f['coarse_hits'])
    print (f['pixel_hits'])
    print (f['meta'].dtype)
    print (f['coarse_hits'].dtype)
    print (f['pixel_hits'].dtype)

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
    
    for event_index in range(10):
        meta_mask = f['meta']['event id'] == event_index
        event_meta = f['meta'][meta_mask]
        
        coarse_hits_mask = f['coarse_hits']['event id'] == event_index
        event_coarse_hits = f['coarse_hits'][coarse_hits_mask]
        
        pixel_hits_mask = f['pixel_hits']['event id'] == event_index
        event_pixel_hits = f['pixel_hits'][pixel_hits_mask]

        # labels.resize(labels.shape[0] + 1)
        # labels[-1] = 
        these_labels = np.array([(event_meta['vertex x'],
                                  event_meta['vertex y'],
                                  event_meta['vertex z'],
                                  event_meta['deposited charge'],
                                  )],
                                dtype = label_dtype)
        labels = np.concatenate((labels, these_labels))
        print ("labels", these_labels)
        print (event_meta)
        print (event_coarse_hits)
        print (event_pixel_hits)

        event_metrics = get_simple_metrics(event_pixel_hits,
                                           event_coarse_hits)
        these_metrics = np.array([event_metrics],
                                 dtype = metric_dtype)
        metrics = np.concatenate((metrics, these_metrics))
        print ("metrics", these_metrics)
                                   
        
        # simple_estimate = simple_point_source_position_estimate(event_pixel_hits,
        #                                                         event_coarse_hits)

        # # estimate = point_source_position_estimate(event_pixel_hits,
        # #                                           event_coarse_hits,
        # #                                           detector_model)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection = '3d')
        
        # # plot_coarse_hits(ax,
        # #                  event_coarse_hits['tile x'],
        # #                  event_coarse_hits['tile y'],
        # #                  event_coarse_hits['hit z'],
        # #                  readout_config)
        # plot_pixels(ax,
        #             event_pixel_hits['pixel x'],
        #             event_pixel_hits['pixel y'],
        #             event_pixel_hits['hit z'],
        #             readout_config)
        
        # ax.scatter(simple_estimate[0],
        #            simple_estimate[1],
        #            simple_estimate[2],
        #            marker = '*',
        #            color = 'red',
        #            label = 'Initial Guess',
        #            )

        # # ax.scatter(estimate[0],
        # #            estimate[1],
        # #            estimate[2],
        # #            marker = '*',
        # #            color = 'yellow',
        # #            label = 'Refined Guess',
        # #            )
        
        # ax.scatter(event_meta['vertex x'],
        #            event_meta['vertex y'],
        #            event_meta['vertex z'],
        #            marker = '*',
        #            color = 'green',
        #            label = 'True Point Source Position',
        #            )

        # ax.legend()

        # plt.show()
    print (labels.shape)
    print (metrics.shape)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('input_edepsim_file',
    #                     type = str,
    #                     help = 'input file from which to read and simulate an event')
    # parser.add_argument('-e', '--event_index',
    #                     type = int,
    #                     default = 5,
    #                     help = 'index of the event within the input file to be simulated')
    # parser.add_argument('-o', '--output_file',
    #                     type = str,
    #                     default = "",
    #                     help = 'output hdf5 file to store coarse tile and pixel measurements')

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

