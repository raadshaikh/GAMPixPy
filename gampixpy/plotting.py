import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class EventDisplay:
    MAX_POINTS_PLOTTED = 1e4
    def __init__(self, track):
        self.track_object = track
        self.init_fig()
        
    def init_fig(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection = '3d')

    def show(self):
        plt.show()

    def plot_raw_track(self):

        n_points = self.track_object.raw_track['position'].shape[0]
        if n_points > self.MAX_POINTS_PLOTTED:
            reduction_factor = math.ceil(n_points/self.MAX_POINTS_PLOTTED)
            self.ax.scatter(self.track_object.raw_track['position'][::reduction_factor,0],
                            self.track_object.raw_track['position'][::reduction_factor,1],
                            self.track_object.raw_track['position'][::reduction_factor,2],
                            c = np.log(self.track_object.raw_track['charge'][::reduction_factor]),
                            )
        else:
            self.ax.scatter(self.track_object.raw_track['position'][:,0],
                            self.track_object.raw_track['position'][:,1],
                            self.track_object.raw_track['position'][:,2],
                            c = np.log(self.track_object.raw_track['charge'][:]),
                            )

        self.ax.set_xlabel(r'x (transverse)')
        self.ax.set_ylabel(r'y (transverse)')
        self.ax.set_zlabel(r'z (drift)')
            
    def plot_drifted_track(self):

        n_points = self.track_object.drifted_track['position'].shape[0]
        if n_points > self.MAX_POINTS_PLOTTED:
            reduction_factor = math.ceil(n_points/self.MAX_POINTS_PLOTTED)
            self.ax.scatter(self.track_object.drifted_track['position'][::reduction_factor,0],
                            self.track_object.drifted_track['position'][::reduction_factor,1],
                            self.track_object.drifted_track['position'][::reduction_factor,2],
                            c = np.log(self.track_object.drifted_track['charge'][::reduction_factor]),
                            )
        else:
            self.ax.scatter(self.track_object.drifted_track['position'][:,0],
                            self.track_object.drifted_track['position'][:,1],
                            self.track_object.drifted_track['position'][:,2],
                            c = np.log(self.track_object.drifted_track['charge'][:]),
                            )

        self.ax.set_xlabel(r'x (transverse)')
        self.ax.set_ylabel(r'y (transverse)')
        self.ax.set_zlabel(r'z (drift)')
            
    def plot_coarse_tile_measurement(self, readout_config):
        # TODO: implement!

        for this_hit in self.track_object.coarse_tiles_samples:
            cell_center_xy = this_hit.coarse_cell_id
            cell_center_z = this_hit.coarse_measurement_time
            cell_measurement = this_hit.coarse_cell_measurement

            x_bounds = [cell_center_xy[0] - 0.5*readout_config['coarse_tiles']['pitch'],
                        cell_center_xy[0] + 0.5*readout_config['coarse_tiles']['pitch']]
            y_bounds = [cell_center_xy[1] - 0.5*readout_config['coarse_tiles']['pitch'],
                        cell_center_xy[1] + 0.5*readout_config['coarse_tiles']['pitch']]
            z_bounds = [cell_center_z,
                        cell_center_z + readout_config['coarse_tiles']['z_bin_width']*readout_config['coarse_tiles']['integration_length']]
            
            
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
            # self.ax.scatter(this_hit.coarse_cell_id[0],
            #                 this_hit.coarse_cell_id[1],
            #                 this_hit.coarse_measurement_time,
            #                 c = this_hit.coarse_cell_measurement,
            #                 )
            self.ax.add_collection3d(Poly3DCollection(faces, facecolors = 'cyan', linewidths=1, edgecolors = 'k', alpha = 0.25))
            
        self.ax.set_xlabel(r'x (transverse)')
        self.ax.set_ylabel(r'y (transverse)')
        self.ax.set_zlabel(r'z (drift)')
            
    def plot_pixel_measurement(self, readout_config):
        
        for this_hit in self.track_object.pixel_samples:
            cell_center_xy = this_hit.pixel_id
            cell_center_z = this_hit.hit_timestamp
            cell_measurement = this_hit.hit_measurement

            x_bounds = [cell_center_xy[0] - 0.5*readout_config['pixels']['pitch'],
                        cell_center_xy[0] + 0.5*readout_config['pixels']['pitch']]
            y_bounds = [cell_center_xy[1] - 0.5*readout_config['pixels']['pitch'],
                        cell_center_xy[1] + 0.5*readout_config['pixels']['pitch']]
            z_bounds = [cell_center_z,
                        cell_center_z + readout_config['pixels']['z_bin_width']*readout_config['pixels']['integration_length']]
            
            
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
            # self.ax.scatter(this_hit.coarse_cell_id[0],
            #                 this_hit.coarse_cell_id[1],
            #                 this_hit.coarse_measurement_time,
            #                 c = this_hit.coarse_cell_measurement,
            #                 )
            self.ax.add_collection3d(Poly3DCollection(faces, facecolors = 'orange', linewidths=1, edgecolors = 'k', alpha = 0.25))
            
        self.ax.set_xlabel(r'x (transverse)')
        self.ax.set_ylabel(r'y (transverse)')
        self.ax.set_zlabel(r'z (drift)')
