import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure                      
from matplotlib.backends.backend_agg import FigureCanvasAgg

import cStringIO
from PIL import Image
import time

import os
import errno

def cart2sph(X, Y, Z):
    """Converts cartesian to spherical coordinates.
    Works on matrices so we can pass in e.g. X with rows of len 3 for polygons."""
    
    XY = X**2 + Y**2
    TH = np.arctan2(Y, X) #theta: azimuth
    PHI = np.arctan2(Z, np.sqrt(XY)) #phi: elevation from XY plane up
    R = np.sqrt(XY + Z**2) #r
    
    return (TH, PHI, R)


def pi2pi(theta):
    """Constrains value to lie between -pi and pi."""
    return np.mod(theta + np.pi, 2 * np.pi) - np.pi


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

            
default_ground_colour = np.array([229.0, 183.0, 90.0]) / 255


class WorldViewer(object):
    """Generates views for a particular world and ant config."""

    dpi = 100 # Set to this as it makes it easy to specify pixel dimensions
    
    def __init__(self, X, Y, Z, c,
                 hfov_deg=296.0,
                 v_max=np.pi/3,
                 v_min=-np.pi/12,
                 resolution=4,
                 world_name=None,
                 data_folder='data/antworlds',
                 ground_colour=default_ground_colour,
                 sky_colour='cyan',
                 grass_colour=(0,1,0,1)
                 ):
        
        
            
        if world_name is None:
            self.world_name = time.strftime("%Y%m%d-%H%M%S")
        else:
            self.world_name = world_name
        
        self.data_folder = data_folder
        
        self.ground_colour = ground_colour
        self.sky_colour = sky_colour
        self.grass_cmap = LinearSegmentedColormap.from_list('mycmap', [(0, (0,0,0,1)),
                                                                       (1, grass_colour)])
        
        self.X = X
        self.Y = Y
        self.Z = Z
        if c is None:
            c = np.ones(Z.shape[0]) * 0.5
        self.c = c
        
        # Set field of view properties
        self.hfov_deg = hfov_deg
        self.hfov = np.deg2rad(hfov_deg)
        self.h_min = -self.hfov / 2
        self.h_max = self.hfov / 2
        
        self.v_max = v_max
        self.v_min = v_min
        self.vfov = v_max - v_min
        self.vfov_deg = np.rad2deg(self.vfov) # Todo: make this an attribute.
        
        self.resolution = resolution
        
        # Calculate image size
        self.im_width, self.im_height = self.calc_image_size()
        
        # Create ground points
        self.ground = self.create_ground()
        
        # Create figure and axis that can be re-used
        self.fig = Figure(frameon=False,
                          figsize=(self.im_width, self.im_height))
        
        self.ax = self.fig.add_axes([0., 0., 1., 1.])
        self.ax.set_ylim(self.v_min, self.v_max)
        self.ax.set_xlim(self.h_min, self.h_max)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.set_axis_bgcolor(self.sky_colour)

        self.canvas = FigureCanvasAgg(self.fig)
        
        
    def calc_image_size(self):
        """Calculates the image size based on FOV and ant eye resolution."""
        image_ratio = self.vfov / self.hfov
        
        h_pixels = self.hfov_deg / self.resolution
        v_pixels = h_pixels * image_ratio
        
        im_width = h_pixels / self.dpi
        im_height = v_pixels / self.dpi
        return (im_width, im_height)
    
    
    def create_ground(self):
        """Creates a rectangular patch for the ground."""                
        ground_verts = [[(self.h_min, self.v_min),
                        (self.h_max, self.v_min),
                        (self.h_max, 0),
                        (self.h_min, 0)]]

        g = PolyCollection(ground_verts,
                           facecolor=self.ground_colour,
                           edgecolor='none')
        return g
        
        
    def fix_grass(self, TH, PHI, R, colours):
        """Ensures that grass spanning over -pi and +pi behaves correctly"""

        # Find grasses that span large angle (over 180 deg)
        ind = (np.max(TH, axis=1) - np.min(TH, axis=1)) > np.pi

        # Duplicate some blades so we can plot grasses that span
        # the crossover point when using wide angles.
        TH_ext = np.vstack((TH, np.mod(TH[ind, :]-2*np.pi, -2*np.pi)))

        n_blades = np.sum(ind)
        padded_ind = np.lib.pad(ind, (0,n_blades), 'constant')
        TH_ext[padded_ind, :] = np.mod(TH[ind,:] + 2*np.pi, 2*np.pi)

        PHI_ext = np.vstack((PHI, PHI[ind, :]))
        R_ext = np.vstack((R, R[ind, :]))
        colours_ext = np.concatenate((colours, colours[ind]))
        
        return TH_ext, PHI_ext, R_ext, colours_ext
        
    
    def create_plot(self, TH, PHI, colours):
        """Generates the figure by placing grass patches."""
        
        # Clear the axis
        self.ax.cla()
        
        # Add the ground
        self.ax.add_collection(self.ground)
        
        # Add the grass        
        grass_verts = np.dstack((TH, PHI))
        
        p = PolyCollection(grass_verts,
                           array=colours,
                           cmap=self.grass_cmap,
                           edgecolors='none')
        self.ax.add_collection(p)
        
        
    def create_image(self):
        """Generates an image based on the current state of self.fig"""
        buf = cStringIO.StringIO()
                
        self.fig.savefig(buf,
                         format='png',
                         pad_inches=0,
                         dpi=self.dpi)
        
        buf.seek(0)  # rewind the data
        im = Image.open(buf)
        im_array = np.asarray(im)[:,:,0:3]
        
        return im_array
    
    
    #def create_image2(self):
    #    """Generates an image based on the current state of self.fig"""
    #    self.canvas.draw()
    #    data = np.fromstring(self.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    #    data = data.reshape(self.canvas.get_width_height()[::-1] + (3,))
    #    return data
    
    
    # CHange this to an attribute?
    def file_path(self):
        """Returns the path to store images"""
        return self.data_folder + '/' + self.world_name + '/'
    
    
    def create_file(self, filename):
        """Create the png file to save view into."""
        
        
        make_sure_path_exists(self.file_path())
        
        self.fig.savefig(self.file_path()+filename,
                         format='png',
                         pad_inches=0,
                         dpi=self.dpi)
        return
    
        
    def get_view(self, x, y, th, z=0.01):
        """Generates the current view and outputs as numpy array"""
        
        # Get view relative to current location
        TH, PHI, R = cart2sph(self.X-x, self.Y-y, np.abs(self.Z)-z)
        
        # Convert to range + and - pi
        TH_rel = pi2pi(TH - th)
        
        # Fix grasses that fall outside this range
        TH2, PHI2, R2, colours2 = self.fix_grass(TH_rel, PHI, R, self.c)
        
        # Get indices sorted by descending order of distance from location
        sorted_idxs = np.argsort(np.mean(R2, axis=1))[::-1]
        
        # Create Matplotlib plot
        self.create_plot(TH2[sorted_idxs,:],
                         PHI2[sorted_idxs,:],
                         colours2[sorted_idxs])
        
        # Output as numpy array
        im_array = self.create_image()
        
        return im_array

    
    # TODO: lots of duplicate code here ... refactor
    def save_view(self, x, y, th, z=0.01, xoffset=0, yoffset=0):
        """Generates the current view and outputs to a png file"""
        
        filename = "{:0>5.2f}x_{:0>5.2f}y_{:0>5.2f}z_{:.2f}th_{:0>3d}fov.png".format(x+xoffset,
                                                                                     y+yoffset,
                                                                                     z,
                                                                                     th,
                                                                                     int(self.hfov_deg))
        
        # Only do all this stuff if we haven't already done it for the same location.
        if os.path.isfile(self.file_path()+filename) is False:
            
            # Get view relative to current location
            TH, PHI, R = cart2sph(self.X-x, self.Y-y, np.abs(self.Z)-z)

            # Convert to range + and - pi
            TH_rel = pi2pi(TH - th)

            # Fix grasses that fall outside this range
            TH2, PHI2, R2, colours2 = self.fix_grass(TH_rel, PHI, R, self.c)

            # Get indices sorted by descending order of distance from location
            sorted_idxs = np.argsort(np.mean(R2, axis=1))[::-1]

            # Create Matplotlib plot
            self.create_plot(TH2[sorted_idxs,:],
                             PHI2[sorted_idxs,:],
                             colours2[sorted_idxs])


            # Save as png
            self.create_file(filename)
        
        return