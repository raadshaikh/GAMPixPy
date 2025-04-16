import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import SLACplots

class EventDisplay:
    MAX_POINTS_PLOTTED = 1e3
    def __init__(self, track):
        self.track_object = track
        self.init_fig()

        # self.coarse_tile_hit_kwargs = dict(facecolors = 'cyan',
        self.coarse_tile_hit_kwargs = dict(facecolors=SLACplots.stanford.palo_verde,
                                           linewidths=1,
                                           edgecolors=SLACplots.SLACblue,
                                           alpha = 0.25)
        self.pixel_hit_kwargs = dict(facecolors=SLACplots.stanford.illuminating,
                                     linewidths=1,
                                     edgecolors=SLACplots.stanford.illuminating,
                                     alpha = 0.5)
        
    def init_fig(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection = '3d')

    def remove_guidelines(self):
        self.ax.axis('off')
        
    def equal_aspect(self):
        extents = np.array([getattr(self.ax, 'get_{}lim'.format(dim))() for dim in 'xy'])
        sz = extents[:,1] - extents[:,0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize/2
        for ctr, dim in zip(centers, 'xy'):
            getattr(self.ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
            
    def show(self):
        self.equal_aspect()
        plt.show()
        
    def save(self, outfile):
        self.equal_aspect()
        self.fig.savefig(outfile)

    def draw_box(self, cell_center_xy, cell_center_z, cell_pitch, cell_height, **kwargs):
        x_bounds = [cell_center_xy[0] - 0.5*cell_pitch,
                    cell_center_xy[0] + 0.5*cell_pitch]
        y_bounds = [cell_center_xy[1] - 0.5*cell_pitch,
                    cell_center_xy[1] + 0.5*cell_pitch]
        z_bounds = [cell_center_z,
                    cell_center_z + cell_height]
            
        bottom_face = np.array([[x_bounds[0], x_bounds[0],
                                 x_bounds[1], x_bounds[1], x_bounds[0]],
                                [y_bounds[0], y_bounds[1],
                                 y_bounds[1], y_bounds[0], y_bounds[0]],
                                [z_bounds[0], z_bounds[0],
                                 z_bounds[0], z_bounds[0], z_bounds[0]]]).T
        top_face = np.array([[x_bounds[0], x_bounds[0],
                              x_bounds[1], x_bounds[1], x_bounds[0]],
                             [y_bounds[0], y_bounds[1],
                              y_bounds[1], y_bounds[0], y_bounds[0]],
                             [z_bounds[1], z_bounds[1],
                              z_bounds[1], z_bounds[1], z_bounds[1]]]).T
        left_face = np.array([[x_bounds[0], x_bounds[0],
                               x_bounds[0], x_bounds[0], x_bounds[0]],
                              [y_bounds[0], y_bounds[1],
                               y_bounds[1], y_bounds[0], y_bounds[0]],
                              [z_bounds[0], z_bounds[0],
                               z_bounds[1], z_bounds[1], z_bounds[0]]]).T
        right_face = np.array([[x_bounds[1], x_bounds[1],
                                x_bounds[1], x_bounds[1], x_bounds[1]],
                               [y_bounds[0], y_bounds[1],
                                y_bounds[1], y_bounds[0], y_bounds[0]],
                               [z_bounds[0], z_bounds[0],
                                z_bounds[1], z_bounds[1], z_bounds[0]]]).T
        back_face = np.array([[x_bounds[0], x_bounds[1],
                               x_bounds[1], x_bounds[0], x_bounds[0]],
                              [y_bounds[0], y_bounds[0],
                               y_bounds[0], y_bounds[0], y_bounds[0]],
                              [z_bounds[0], z_bounds[0],
                               z_bounds[1], z_bounds[1], z_bounds[0]]]).T
        front_face = np.array([[x_bounds[0], x_bounds[1],
                                x_bounds[1], x_bounds[0], x_bounds[0]],
                               [y_bounds[1], y_bounds[1],
                                y_bounds[1], y_bounds[1], y_bounds[1]],
                               [z_bounds[0], z_bounds[0],
                                z_bounds[1], z_bounds[1], z_bounds[0]]]).T
        faces = [bottom_face,
                 top_face,
                 left_face,
                 right_face,
                 back_face,
                 front_face,
                 ]
        
        self.ax.add_collection3d(Poly3DCollection(faces, **kwargs))            

    def plot_raw_track(self, **kwargs):

        n_points = self.track_object.raw_track['position'].shape[0]
        if n_points > self.MAX_POINTS_PLOTTED:
            reduction_factor = math.ceil(n_points/self.MAX_POINTS_PLOTTED)
            xs = self.track_object.raw_track['position'][::reduction_factor,0]
            ys = self.track_object.raw_track['position'][::reduction_factor,1]
            zs = self.track_object.raw_track['position'][::reduction_factor,2]
            colors = np.log(self.track_object.raw_track['charge'][::reduction_factor])
        else:
            xs = self.track_object.raw_track['position'][:,0],
            ys = self.track_object.raw_track['position'][:,1],
            zs = self.track_object.raw_track['position'][:,2],
            colors = np.log(self.track_object.raw_track['charge'][:])
            
        self.ax.scatter(xs, ys, zs,
                        c = colors,
                        **kwargs,
                        )

        self.ax.set_xlabel(r'x (transverse) [cm]')
        self.ax.set_ylabel(r'y (transverse) [cm]')
        self.ax.set_zlabel(r'z (drift) [cm]')
            
    def plot_drifted_track(self, **kwargs):

        n_points = self.track_object.drifted_track['position'].shape[0]
        if n_points > self.MAX_POINTS_PLOTTED:
            reduction_factor = math.ceil(n_points/self.MAX_POINTS_PLOTTED)
            xs = self.track_object.drifted_track['position'][::reduction_factor,0]
            ys = self.track_object.drifted_track['position'][::reduction_factor,1]
            zs = self.track_object.drifted_track['position'][::reduction_factor,2]
            colors = np.log(self.track_object.drifted_track['charge'][::reduction_factor])
        else:
            xs = self.track_object.drifted_track['position'][:,0]
            ys = self.track_object.drifted_track['position'][:,1]
            zs = self.track_object.drifted_track['position'][:,2]
            colors = np.log(self.track_object.drifted_track['charge'][:])
            
        self.ax.scatter(xs, ys, zs,
                        c = colors,
                        **kwargs,
                        )

        self.ax.set_xlabel(r'x (transverse) [cm]')
        self.ax.set_ylabel(r'y (transverse) [cm]')
        self.ax.set_zlabel(r'z (drift) [cm]')

    def plot_drifted_track_timeline(self, **kwargs):

        n_points = self.track_object.drifted_track['position'].shape[0]
        if n_points > self.MAX_POINTS_PLOTTED:
            reduction_factor = math.ceil(n_points/self.MAX_POINTS_PLOTTED)
            xs = self.track_object.drifted_track['position'][::reduction_factor,0]
            ys = self.track_object.drifted_track['position'][::reduction_factor,1]
            zs = self.track_object.drifted_track['times'][::reduction_factor]
            colors = np.log(self.track_object.drifted_track['charge'][::reduction_factor])
        else:
            xs = self.track_object.drifted_track['position'][:,0]
            ys = self.track_object.drifted_track['position'][:,1]
            zs = self.track_object.drifted_track['times'][:]
            colors = np.log(self.track_object.drifted_track['charge'][:])
                        

        self.ax.scatter(xs, ys, zs,
                        c = colors,
                        **kwargs,
                        )

        self.ax.set_xlabel(r'x (transverse) [cm]')
        self.ax.set_ylabel(r'y (transverse) [cm]')
        self.ax.set_zlabel(r'z (drift) [cm]')
        
    def plot_coarse_tile_measurement(self, readout_config):
        for this_hit in self.track_object.coarse_tiles_samples:
            cell_center_xy = this_hit.coarse_cell_pos
            # cell_center_z = this_hit.coarse_measurement_time
            cell_center_t = this_hit.coarse_measurement_time
            v = 1.6e5
            cell_center_z = cell_center_t*v
            cell_hit_length = v*readout_config['coarse_tiles']['clock_interval']*readout_config['coarse_tiles']['integration_length']
            cell_measurement = this_hit.coarse_cell_measurement

            self.draw_box(cell_center_xy, cell_center_z,
                          readout_config['coarse_tiles']['pitch'],
                          cell_hit_length,
                          **self.coarse_tile_hit_kwargs)
            
        self.ax.set_xlabel(r'x (transverse) [cm]')
        self.ax.set_ylabel(r'y (transverse) [cm]')
        self.ax.set_zlabel(r'z (drift) [cm]')

    def plot_coarse_tile_measurement_timeline(self, readout_config):
        xlim = []
        ylim = []
        zlim = []
        
        for this_hit in self.track_object.coarse_tiles_samples:
            cell_center_xy = this_hit.coarse_cell_pos
            # cell_center_z = this_hit.coarse_measurement_time
            cell_trigger_t = this_hit.coarse_measurement_time
            cell_measurement = this_hit.coarse_cell_measurement
            cell_hit_length = readout_config['coarse_tiles']['clock_interval']*readout_config['coarse_tiles']['integration_length']

            self.draw_box(cell_center_xy, cell_trigger_t,
                          readout_config['coarse_tiles']['pitch'],
                          cell_hit_length,
                          **self.coarse_tile_hit_kwargs)
            
        self.ax.set_xlabel(r'x (transverse) [cm]')
        self.ax.set_ylabel(r'y (transverse) [cm]')
        self.ax.set_zlabel(r'Arrival Time [us]')
        
    def plot_pixel_measurement(self, readout_config):
        
        for this_hit in self.track_object.pixel_samples:
            cell_center_xy = this_hit.pixel_pos
            cell_center_t = this_hit.hit_timestamp
            v = 1.6e5
            cell_center_z = cell_center_t*v
            cell_measurement = this_hit.hit_measurement
            cell_hit_length = v*readout_config['pixels']['clock_interval']*readout_config['pixels']['integration_length']
            
            self.draw_box(cell_center_xy, cell_center_z,
                          readout_config['pixels']['pitch'],
                          cell_hit_length,
                          **self.pixel_hit_kwargs)
            
        self.ax.set_xlabel(r'x (transverse) [cm]')
        self.ax.set_ylabel(r'y (transverse) [cm]')
        self.ax.set_zlabel(r'z (drift) [cm]')

    def plot_pixel_measurement_timeline(self, readout_config):
        
        for this_hit in self.track_object.pixel_samples:
            cell_center_xy = this_hit.pixel_pos
            # cell_center_z = this_hit.hit_timestamp
            cell_trigger_t = this_hit.hit_timestamp
            cell_measurement = this_hit.hit_measurement
            cell_hit_length = readout_config['pixels']['clock_interval']*readout_config['pixels']['integration_length']
            
            self.draw_box(cell_center_xy, cell_trigger_t,
                          readout_config['pixels']['pitch'],
                          cell_hit_length,
                          **self.pixel_hit_kwargs)
            
        self.ax.set_xlabel(r'x (transverse) [cm]')
        self.ax.set_ylabel(r'y (transverse) [cm]')
        self.ax.set_zlabel(r'arrival time [us]')
