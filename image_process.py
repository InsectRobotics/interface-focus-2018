# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:46:36 2015

@author: tomish
"""
import os
import os.path
import cv2
import numpy as np
import numpy.ma as ma # For masking
import scipy as sp
import scipy.spatial.distance
from scipy.ndimage.filters import gaussian_filter1d
from scipy import interpolate
from scipy.stats import norm

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import skimage.morphology as morp
from skimage.filters import rank
from PIL import Image
from mahotas.features import zernike_moments

import pyshtools as shtools # Need to install this!!

from image_set import ImgSet
import fourier_descriptors as fd
import spherical_harmonics as sh
import zernike

# Added for Dario stuff NOT SURE I LIKE THIS! :(
#from hsc_wrapper import HSC_wrapper
#import hsc_wrapper as hsc  #uncomment this when needed but really should refactor (ALOT!)

class ImgProcess:
    NONE = 1
    FOURIER_DESCRIPTOR = 2
    ZERNIKE_MOMENTS = 3
    HU_MOMENTS = 4
    PRINCIPLE_COMPONENT_ANALYSIS = 5
    IMAGE_CHANGE = 5
    IMAGE_CHANGED = 7 #Boolean
    RESIZE = 8
    
    cache_results = True # For now just always cache results    
    
    #refactor by doing dict lookup rather than enum
    processes = {}
    processes[2] = 'fd'
    processes[3] = 'zernike'
    processes[4] = 'loghu'
    processes[5] = 'pca'
    processes[6] = 'im_change'
    processes[7] = 'im_changed'
    processes[8] = 'resize'
    
    
    @classmethod
    def process(cls, img_set, process_type):
        
        # Just return if no processing needed
        if process_type == cls.NONE:
            return img_set
        
        # See if we already did this process before
        processed_name = cls.get_processed_name(img_set, process_type)
        
        cache_file = cls.get_cache_file(img_set.data_folder, processed_name)
        loaded_set = ImgSet.load_from_cache(cache_file)
        
        if loaded_set is not None:
            print "Loading cache file: " + cache_file
            return loaded_set
        
        else: # If not previously... calculate
            (M, h, w) = cls.process_img_set(img_set, process_type)
            
            # If we resized also include the new size
            if cls.processes[process_type] == 'resize':
                name = '%s_%s%s' % (img_set.name, cls.processes[process_type], w)
            else:
                name = '%s_%s' % (img_set.name, cls.processes[process_type])
            processed_set = ImgSet(name=name, M=M, w=w, h=h)
            
            # Cache results
            if cls.cache_results:
                processed_set.save()
            
            return processed_set
        
    
    @classmethod
    def process_img_set(cls, img_set, process_type):
        #if process_type == 1:
        #    print "Unprocessed image set"
        #    return img_set.M
            
        if process_type == cls.FOURIER_DESCRIPTOR:
            print "Calculating Fourier descriptors"
            return cls.get_fds(img_set)
            
        elif process_type == cls.ZERNIKE_MOMENTS:
            return cls.get_zms(img_set)
            
        elif process_type == cls.HU_MOMENTS:
            print "Calculating Hu Moments"
            return (cls.get_moments(img_set, 2), 7, 1)
            
        elif process_type == cls.PRINCIPLE_COMPONENT_ANALYSIS:
            return cls.do_pca(img_set.M.T)
            
        elif process_type == cls.IMAGE_CHANGE:
            return cls.pixel_changes(img_set.M.T)
            
        elif process_type == cls.IMAGE_CHANGED:
            return cls.pixel_changed(img_set.M)
            
        elif process_type == cls.RESIZE:
            return cls.resize(img_set)
    
    
    #TODO: Use cumsum(range)?
    @classmethod
    def triangular_num(cls, n):
        """Helper function: 1+2+3+4....+n"""
        return sum([i for i in range(n+1)])
    
    @classmethod
    def get_processed_name(cls, img_set, process):
        return '%s_%s' % (img_set.name, cls.processes[process])
        
    
    @classmethod
    def get_cache_file(cls, data_folder, processed_name):
        return '%s/%s.npz' % (data_folder, processed_name)
    
    
    #@classmethod
    #def file_exists(cls, cache_file):
    #    if os.path.isfile(cache_file):
    #        return cache_file
    #    else:
    #        return None


    @classmethod
    def resize(cls, img_set, w=30, interp='nearest'):
        """Resize images"""
        M_resized = np.zeros([w**2, len(img_set)])
        for i, img in enumerate(img_set):
            img_resized = sp.misc.imresize(img, (w,w), interp)
            M_resized[:,i] = np.reshape(img_resized, -1)
        return (M_resized, w, w)
        
    
    
    @classmethod
    def find_contours(cls, I):
        """ Finds the contours of largest object"""
        contours = []
        _, contours, hierarchy = cv2.findContours(
            I.copy(),
            cv2.RETR_LIST,  # I think this is the fastest
            cv2.CHAIN_APPROX_NONE,
            contours)
        # Return the contour of the largest object
        contour_sizes = [c.size for c in contours]
        contour = contours[contour_sizes.index(max(contour_sizes))]
        return contour


    @classmethod
    def get_moments(cls, img_set, method=1):
        """ Gets Hu Moments for a set of images"""
        num_moments = 7
        moments = np.empty([num_moments, len(img_set)])
        for i, img in enumerate(img_set):
            moments[:,i] = cls.calc_moment(img)
        # Methods copied from opencv.modules.imgproc.src.matchcontours.cpp
        if method == 1:
            return 1./(np.sign(moments) * np.log10(np.abs(moments)))
        else:
            return np.sign(moments) * np.log10(np.abs(moments))
 
    
    @classmethod
    def get_fds(cls, img_set, num_descriptors=100):
        """gets FD for a set of images. Returns the magnitude"""
        fd = np.empty([num_descriptors, len(img_set)], dtype=complex)
        for i, img in enumerate(img_set):
            fd[:,i] = cls.calc_fd(img, num_descriptors)
        return (np.absolute(fd), num_descriptors, 1)
    
    
    @classmethod
    def get_zms(cls, img_set, polynomial=10):
        radius = img_set.w / 2.0 # Can this be bigger than the image?
        cm = (img_set.w/2.0-0.5, img_set.h/2.0-0.5) # Centre of mass
        num_moments = cls.triangular_num((polynomial+2) / 2) + cls.triangular_num((polynomial+1) / 2) # times something?
        
        zms = np.empty([num_moments, len(img_set)], dtype=complex)
        for i, img in enumerate(img_set):
            zms[:,i] = zernike_moments(img, radius, polynomial, cm=cm)
        return (zms, num_moments, 1)
    
    
    
    @classmethod
    def match_shapes(cls, hu1, hu2):
        #hu1_mod = 1 ./ (np.sign(hu1) * np.log10(np.abs(hu1))
        #hu2_mod = 1 ./ (np.sign(hu1) * np.log10(np.abs(hu1))
        #return np.abs(hu2_mod - hu1_mod)
        pass
    
    
    @classmethod
    def biggest_area(cls, I):
        X = np.zeros(I.shape, dtype='uint8')
        contours = cls.find_contours(I)
        return cv2.drawContours(X, [contours], -1, 255, -1 )
    
    
    @classmethod
    def calc_fd(cls, I, num_descriptors=100):
        """Calculate Fourier descriptors of a binary image"""
        contours = cls.find_contours(I)
        descriptors = fd.find_descriptors(contours)
        return descriptors[0:num_descriptors]
        
        
    @classmethod
    def calc_moment(cls, I):
        """Calculate the Hu moments of a binary image"""
        # Get different answers depending on input to moments
        contours = cls.find_contours(I)
        moments = cv2.HuMoments(cv2.moments(contours)).flatten()
        #moments = cv2.HuMoments(cv2.moments(I)).flatten()
        return moments
    
    
    
    @staticmethod
    def do_pca(X, n_components=1000):
        pca = PCA()
        return pca.transform(X)
        
        
    @staticmethod        
    def pixel_changes(X):
        
        #return np.logical_xor(X[1:], X[:-1])
        return (X[1:].astype('i2') - X[:-1].astype('i2')).T
    
    
    @staticmethod        
    def pixel_changed(X):
        return np.logical_xor(X[1:], X[:-1])
    
    
    
    


"""ABOVE HERE IS LEGACY STUFF"""

def sad(I1, I2):
    """Returns the summed absolute difference between two grayscale images."""
    # Need to use signed ints to get difference to work
    return np.sum(np.abs(I1 - I2))


def wraparound_sad(I1, I2, x_res=None):
    """Returns SAD between two grayscale images, at various rotations."""
    w = I1.shape[1]
    
    if x_res is None:
        x_res = w
        
    rotations = np.rint(np.linspace(0, w, num=x_res, endpoint=False)).astype('uint32')
    
    return np.array([sad(I1, np.roll(I2, r, axis=1)) for r in rotations])


def calc_distance_sad_slow(S1, S2, x_res=None):
    """Compare two image sets. Rotate each image to get best match"""
    D = np.empty([len(S1), len(S2)])
    
    
    #Need to convert to int16 to get the negative stuff working

    for i, I1 in enumerate(S1):
        for j, I2 in enumerate(S2):
            D[i,j] = np.min(wraparound_sad(I1.astype('int16'), I2.astype('int16'), x_res=x_res))
    
    return D
    
    
def calc_distance_sad(S1, S2, x_res=1, metric='cityblock'):
    """Not that fast, but helluva lot faster than the other one"""
    D = np.empty([len(S1), len(S2)])
    
    h = S1.h
    w = S1.w
    
    x = np.arange(0, h*w, h*x_res)
    y = np.arange(h*w)
    xv, yv = np.meshgrid(x,y)
    idxs = np.mod(xv + yv, h*w)
    
    for i in range(len(S1)):
        D[i,:] = np.min(sp.spatial.distance.cdist(S1.M[:,i][idxs].T, S2.M.T, metric=metric), axis=0)
    return D


def local_contrast_enhance(I):
    """Local histogram equalization"""
    amp_factor = (2**16) / np.max(I)

    # Suggested kernel by milford (centre surround type thing)
    #kernel = np.ones([50, 50], dtype='uint8')
    #kernel[12:18, 12:18] = 0
    
    kernel = np.ones([1, 30], dtype='uint8')
    return rank.equalize((I * amp_factor).astype('u2'), selem=kernel)


def rotate_image(I, theta):
    """ Rotates an image by theta degrees, while maintaining same scale and size."""
    rows, cols = I.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
    return cv2.warpAffine(I, M, (cols, rows))


def resize_image(I, w=30, interp='nearest'):
    """Resize image"""
    return sp.misc.imresize(img, (w,w), interp)


def binarize(I):
    """Converts an image to B&W using Otsu's method"""
        
    # If colour, then convert to grayscale
    if len(I.shape) > 2:
        I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
        
    
    
    # Find threshold
    threshold, BW = cv2.threshold(I,
                                 0,
                                 255,
                                 cv2.THRESH_OTSU)
 
    return BW


def binarize_na(I):
    """Dario's method to binarize"""
    threshold, BW = cv2.threshold(I,
                                  0,
                                  255,
                                  cv2.THRESH_OTSU)

    # fit sky to a gaussian and chop after 2 standard deviations
    #print BW
    #print np.max(BW)
    #print np.min(BW)
    #print BW.shape
    #print I.shape
    #print I[BW.reshape(I.shape)==255].shape
    mu, std = norm.fit(I[BW.reshape(I.shape)==255])
    t = mu - 2*std
    return (I>t).astype('int') * 255

    

def binarize_ws(I, perc=0, max_hist_res=1000):
    """Binarise an image using the watershed method described in (Stone 2014)"""
    upper_p = np.percentile(I, 100-perc)
    lower_p = np.percentile(I, perc)
    
    values = np.unique(I)
    values_truncated = values[np.logical_and(values > lower_p, values < upper_p)]
    values_range = np.max(values_truncated) - np.min(values_truncated)
    
    if (values_range > max_hist_res):
        hist_res = max_hist_res
    else:
        hist_res = values_range + 1

    lower = values_truncated[0].astype('float')
    upper = values_truncated[-1].astype('float')
    
    h, null = np.histogram(I, bins=hist_res, range=(lower,upper+1))
    h_gf = gaussian_filter1d(h, sigma=hist_res/20.0)
    hist_values = np.unique(h_gf)
    
    i = 0
    l_idxs = np.array([-1,-1,-1]) # Left hand slope

    while i < hist_values.size and l_idxs.size is not 2:
        a = h_gf > hist_values[i]
        diffs = np.diff(np.hstack([0, a]))
        l_idxs = np.where(diffs == 1)[0] # For some reason where returns a tuple
        i += 1
    
    if l_idxs.size < 2:
        t = np.max(values)
    else:
        r_idxs = np.where(diffs == -1)[0] # Find right slope
        t = lower + (l_idxs[1] + r_idxs[0]) / 2.0 #assumes 8 bit GS image

    BW = (I>t).astype('int') * 255
    return BW


def mask_image(I, mask):
    return ma.masked_array(I, mask=mask)

    
def circular_mask(width, height, radius, centre=None):
    """Creates a black circular mask on white background.

    Args:
      width (int): The mask width.
      height (int): The mask height.
      radius (int): The radius of the cirle
      centre (int, optional): The centre of the circle.
        This might be useful because the parabolic mirror is
        offset from actual image centre.
            
    """
    if not centre:
        centre = (width / 2, height / 2)
    
    mask = np.ones((height, width), np.uint8) * 255
    cv2.circle(mask, centre, radius, 0, -1)
    
    return mask
    
    
def donut_mask(width, height, r_inner, r_outer, centre=None):
    """ Creates a black donut shaped mask.
    
    For use when there is a blind spot in centre of panoramic mirror.
    
    """
    if not centre:
        centre = width / 2, height / 2
    
    mask = circular_mask(width, height, r_outer, centre)
    cv2.circle(mask, centre, r_inner, 255, -1)

    return mask
    
    
def crop_to_square(I):
    """ Currently assumes a grayscale landscape image!!!"""
    # Todo: include RGB images (3 dimensions)
    h = I.shape[0]
    w = I.shape[1]
    pixels_to_crop = (w - h) / 2
    return I[:, pixels_to_crop:w - pixels_to_crop]


# Refactor all these. they are redundant!
def segment(I, mask=None):
    """Binarises a masked or unmasked image"""
    if mask is not None:
        I_masked = mask_image(I, mask)
        I_seg = np.zeros(I.shape, dtype=np.uint8)
        I_seg[mask==0] = binarize(I_masked.compressed()).flatten()
    else:
        I_seg = binarize(I)

    return I_seg


def segment_na(I, mask=None):
    """Binarises a masked or unmasked image"""
    if mask is not None:
        I_masked = mask_image(I, mask)
        I_seg = np.zeros(I.shape, dtype=np.uint8)
        I_seg[mask==0] = binarize_na(I_masked.compressed()).flatten()
    else:
        I_seg = binarize_na(I)

    return I_seg


def segment_ws(I, mask=None):
    """Binarises a masked or unmasked image"""
    if mask is not None:
        I_masked = mask_image(I, mask)
        I_seg = np.zeros(I.shape, dtype=np.uint8)
        I_seg[mask==0] = binarize_ws(I_masked.compressed()).flatten()
    else:
        I_seg = binarize_ws(I)

    return I_seg
    

def find_contours(I):
    """ Finds the contours of largest object"""
    contours_list = []
    
    # CAUTION This is a temporary fix, as syntax is different for OpenCV 2 and 3
    # After OpenCV is out of beta we can insist on using cv2 3+
    if int(cv2.__version__[0]) > 2:
        _, contours_list, hierarchy = cv2.findContours(I.copy(),
                                                  cv2.RETR_LIST,  # I think this is the fastest
                                                  cv2.CHAIN_APPROX_NONE,
                                                  contours_list)
    else:
        contours_list, hierarchy = cv2.findContours(I.copy(),
                                                  cv2.RETR_LIST,  # I think this is the fastest
                                                  cv2.CHAIN_APPROX_NONE,
                                                  contours_list)
        
    return contours_list
    
    
def find_largest_contours(I):
    """ Finds the contours of largest object"""
    contours_list = find_contours(I)
    contour_sizes = [c.size for c in contours_list]
    contours_list = contours_list[contour_sizes.index(max(contour_sizes))]
    contours = contours_list[:, 0, :] # Reformat by removing third dimension
    return contours
    
    
def fill_contours(im_size, contours):
    """Fill in contours to make a binary mask"""
    blank = np.zeros(im_size, dtype='uint8')
    C = cv2.drawContours(blank, [contours], -1, 255, -1) # Contour width of 1 width to fill.
    return C


def biggest_area(I):
    """Returns the biggest filled shape in a B&W image."""
    contours = find_largest_contours(I)
    #X = np.zeros(I.shape, dtype='uint8')
    #return cv2.drawContours(X, [contours], -1, 255, -1 )

    # This is changed to work in opencv 2. Check it still works for opencv 3!!!!
    X = np.zeros(I.shape, dtype='uint8')
    cv2.drawContours(X, [contours], -1, 255, -1 )
    return X


def find_moment(I):
    """Calculate the Hu moments of a binary image"""
    # Get different answers depending on input to moments
    contours = find_contours(I)
    #TODO: check numpy ipython notebook to see what flatten does
    moments = cv2.HuMoments(cv2.moments(contours)).flatten() 
    #moments = cv2.HuMoments(cv2.moments(I)).flatten()
    return moments
        

def calc_distance(set1, set2, metric='euclidean', cache_results=True):
    """Calculates pairwise distance between all images of two ImgSets."""
    cache_file = '%s/%s_%s_%s.npz' % (set1.data_folder,
                                      set1.name,
                                      set2.name,
                                      metric)

    if os.path.isfile(cache_file):
        print "Loading cache file: %s" % cache_file
        data = np.load(cache_file)
        Z = data['Z']
    else:
        Z = sp.spatial.distance.cdist(set1.M.T, set2.M.T, metric)
        if cache_results:
            np.savez_compressed(cache_file, Z=Z)
    return Z


def rgb_to_gray_enhance_blue(I):
    """Convert an image to grayscale using formula from Shabayek 2012"""
    return



class ImgProc(object):
        
    def __init__(self, img_set, cache_results=True):
        
        self.img_set = img_set
        self.cache_results = cache_results
                
        cache_file = self.get_cache_file_name()
        loaded_set = ImgSet.load_from_cache(cache_file)
        
        if loaded_set is not None:
            print "Loading cache file:", cache_file
            self.processed = loaded_set
        else:
            (M, h, w) = self.process_img_set()
            
            processed_set = ImgSet(name=str(self), M=M, w=w, h=h, data_folder=img_set.data_folder)
            
            if self.cache_results:
                print "Saving to :", cache_file
                processed_set.save()
            
            self.processed = processed_set
    
    
    def process_img_set(self):
        print "Just returning the original set"
        return (self.img_set.M, self.img_set.h, self.img_set.w)
        
        
    def get_cache_file_name(self):
        data_folder = self.img_set.data_folder
        return os.path.join(data_folder, str(self) + '.npz')
        
    
    def __str__(self):
        return '%s_%s' % (self.img_set.name, self.name)

    
    
class SkySegProc(ImgProc):
    """Segment an image into sky and ground.
    
    Currently just using Otsu's method.
    
    TODO: add some other thresholding methods to boost performance.

    Attributes:
      name (str): The tag that will be used to save output.
      mask (bool): To mask out areas of the image that shouldn't be
                   used for calculation of the historgram for segmentation
      single_contour (bool): Sometimes we only want a single contour outline,
                   so we can calculate things like Fourier Descriptors.
      
    """
    
    
    
    def __init__(self, img_set, cache_results=True, mask=None, inner_mask=None, single_contour=False, method='otsu', threshold=None):
        
        #Todo: some kind of check to see if the mask if the right size (setter)
        self.name = 'skyseg'
        self.mask = mask
        self.inner_mask = inner_mask
        self.method = method
        self.threshold = threshold
        self.single_contour = single_contour

        if threshold is not None:
            self.name += 'th'
            self.name += str(threshold)
            self.segmenter = self.segment_by_threshold
        else:
            if method == 'otsu':
                self.name += 'otsu'
                self.segmenter = segment
            elif method == 'na':
                self.name += 'na'
                self.segmenter = segment_na
            else:
                self.name += 'ws'
                self.segmenter = segment_ws       

        if single_contour:
            self.name += 'sc'
            
        super(SkySegProc, self).__init__(img_set, cache_results)

    # TODO: Tidy this up. added as last minute fix to set manual thresholds.
    def segment_by_threshold(self, I, mask):
        """Binarises a masked or unmasked image"""
        if mask is not None:
            I_masked = mask_image(I, mask)
            I_seg = np.zeros(I.shape, dtype=np.uint8)
            I_seg[mask==0] = (I_masked.compressed() > self.threshold).astype('uint8').flatten()
        else:
            I_seg = (I > self.threshold).astype('uint8')

        return I_seg * 255

        
    def process_img_set(self):
        """Performs sky segmentation."""
        print "Extracting skyline"
        
        M = np.empty([self.img_set.h * self.img_set.w, len(self.img_set)], dtype='uint8')
        
        for i, img in enumerate(self.img_set):
            I_seg = self.segmenter(img, self.mask)
            
            if self.inner_mask is not None:
                I_seg[self.inner_mask==255] = 255
            
            if self.single_contour:
                I_seg = biggest_area(I_seg)
            M[:,i] = I_seg.flatten()
        return (M, self.img_set.h, self.img_set.w)
    

    
class CropProc(ImgProc):
    """Crop an image to square.
        
    TODO: add some other thresholding methods to boost performance.

    Attributes:
      name (str): The tag that will be used to save output.
      
    """

    def __init__(self, img_set, cache_results=True):
        self.name = 'crop'
        super(CropProc, self).__init__(img_set, cache_results)
    
    
        
    def process_img_set(self):
        """Crops all images to square."""
        print "Cropping to square"
        dtype = self.img_set.M.dtype
        M = np.empty([self.img_set.h * self.img_set.h, len(self.img_set)], dtype=dtype)
        
        for i, img in enumerate(self.img_set):            
            I_cropped = crop_to_square(img)
            M[:,i] = I_cropped.flatten()
        return (M, self.img_set.h, self.img_set.h)
    

class ResizeProc(ImgProc):
    """Resize image by percentage.
        

    Attributes:
      name (str): The tag that will be used to save output.
      
    """

    def __init__(self, img_set, fraction=1.0, cache_results=True):
        self.name = 'resize' + str(fraction)
        self.fraction = fraction
        super(ResizeProc, self).__init__(img_set, cache_results)
    
    
        
    def process_img_set(self):
        """Resize all images to square."""
        print "Resizing to", self.fraction, "of size"
        dtype = self.img_set.M.dtype
        M = np.empty([self.fraction * self.img_set.h * self.fraction * self.img_set.w,
                      len(self.img_set)], dtype=dtype)
        
        for i, img in enumerate(self.img_set):            
            I_resized = sp.misc.imresize(img, self.fraction)
            M[:,i] = I_resized.flatten()
        return (M, self.fraction*self.img_set.h, self.fraction*self.img_set.w)
    
    
class HistProc(ImgProc):
    """Give a intensity histogram.
        

    Attributes:
      name (str): The tag that will be used to save output.
      
    """

    def __init__(self, img_set, cache_results=True):
        self.name = 'hist'
        super(HistProc, self).__init__(img_set, cache_results)

        
    def process_img_set(self):
        """Crops all images to square."""
        print "Cropping to square"
        
        num_bins = 256
        
        M = np.empty([num_bins, len(self.img_set)])
        
        for i, img in enumerate(self.img_set):            
            hist, bins = np.histogram(img.ravel(), num_bins, [0, num_bins])
            M[:,i] = hist
        return (M, num_bins, 1)
    
    
    
class SHProc(ImgProc):
    """Process an image set to Spherical Harmonics.

    This process assumed a B&W sky thresholded image.

    Attributes:
      name (str): The tag that will be used to save output.
      n_max (int): The number of Coefficients to use.
      
    """
        
    def __init__(self, img_set, cache_results=True, n_max=100):
        self.n_max = n_max
        self.name = "sh%s" % n_max

        # First create theta and phi matching current image projection
        w_img = img_set.w
        (theta_masked, phi_masked, self.mask) = sh.generate_spherical_coords(w_img)
        
        # Now create theta and phi for desired projection
        self.w_r = w_img * 2 # The width of the reconstruction
        phi_grid, theta_grid = np.mgrid[0:np.pi:self.w_r * 1j, 0:2 * np.pi:self.w_r * 1j]
        
        # Use nearest neighbours to remap
        self.idx_mapping = sh.create_mapping(theta_masked, phi_masked, theta_grid, phi_grid)
        
        super(SHProc, self).__init__(img_set, cache_results)
        
        
    def process_img_set(self):
        """gets Spherical Harmonics for a set of images. Returns the magnitude"""
        print "Extracting Spherical Harmonics"
        z = np.empty([self.n_max, len(self.img_set)], dtype=float)
        
        inverted_mask = np.invert(self.mask)
        for i, img in enumerate(self.img_set):
            img_vector = img[inverted_mask].reshape(-1)
            values = img_vector[self.idx_mapping]
            img_grid = values.reshape(self.w_r, self.w_r)
            rcoeffs = shtools.SHExpandDH(img_grid, lmax_calc=self.n_max)
            
            z[:,i] = np.linalg.norm(np.linalg.norm(rcoeffs, axis=0), axis=1)[0:self.n_max]
            
            #Attempt with power spectrum
            #z[:,i] = shtools.SHPowerSpectrum(rcoeffs)[0:self.n_max]
            #z[:,i] = shtools.SHPowerSpectrumDensity(rcoeffs)[0:self.n_max]
            
        return (z, self.n_max, 1)
    
class DarioDownSample(ImgProc):


    def __init__(self, img_set, X, Y, cache_results=True, contin=3,
                n_points=2000, n_bands=12, silent=True):

        self.n_bands = n_bands
        self.name = "dariods%s" % n_bands

        rot_par = 1             # 0:XYZ, 1:ZYZ, 2:ZYZ_FAST, 3:XYZ_FAST 4:XYZ_FREE
        resolution_deg = 6.0    # Steps for compass
        tilt_max_deg = 30.0     # Maximum tilt to check

        resolution = np.deg2rad(resolution_deg)
        tilt_max = np.deg2rad(tilt_max_deg)

        rot_dict = hsc.create_rot_par(rot_par, resolution, tilt_max)

        self.hsc_wrapped = hsc.HSC_wrapper(contin,
                                           rot_dict,
                                           n_points,
                                           n_bands,
                                           0,
                                           silent)

        (self.num_angles, theta_sample, phi_sample) = self.hsc_wrapped.get_angles()

        # Bring in the matlab data here for remapping nicely
        indices = np.vstack([np.floor(np.rad2deg(theta_sample)).astype('uint'),
                             np.floor(np.rad2deg(phi_sample)).astype('uint')]).T

        self.xs = X[indices[:,0], indices[:,1]]
        self.ys = Y[indices[:,0], indices[:,1]]

        super(DarioDownSample, self).__init__(img_set, cache_results)

    def panorama_to_surf(self, I):
        surf_img_np = I[np.floor(self.ys).astype('uint'),
                        np.floor(self.xs).astype('uint')]
        surf_img_np = surf_img_np / 255.0 - 0.5
        return surf_img_np.tolist()


    def process_img_set(self):
        z = np.empty([self.num_angles, len(self.img_set)], dtype=float)

        for i, img in enumerate(self.img_set):
            surf_img = self.panorama_to_surf(img)
            z[:,i] = np.array(surf_img)

        return (z, self.num_angles, 1)


class SHProcDarioAS(ImgProc):
    """ Dario's implementation of Spherical Harmonics. Works with panoramic images. Need to refactor and add disk shaped ones too"""


    def __init__(self, img_set, X, Y, cache_results=True, contin=3,
                n_points=2000, n_bands=12, silent=True):
        
        self.n_bands = n_bands
        self.name = "shdas%s" % n_bands

        rot_par = 1             # 0:XYZ, 1:ZYZ, 2:ZYZ_FAST, 3:XYZ_FAST 4:XYZ_FREE
        resolution_deg = 6.0    # Steps for compass
        tilt_max_deg = 30.0     # Maximum tilt to check

        resolution = np.deg2rad(resolution_deg)
        tilt_max = np.deg2rad(tilt_max_deg)

        rot_dict = hsc.create_rot_par(rot_par, resolution, tilt_max)

        self.hsc_wrapped = hsc.HSC_wrapper(contin,
                                           rot_dict,
                                           n_points,
                                           n_bands,
                                           0,
                                           silent)

        (num_angles, theta_sample, phi_sample) = self.hsc_wrapped.get_angles()


        points = (np.arange(X.shape[0]), np.arange(X.shape[1]))

        indices = np.vstack([np.rad2deg(theta_sample),
                             np.rad2deg(phi_sample)]).T
        self.xs = interpolate.interpn(points, X, indices)
        self.ys = interpolate.interpn(points, Y, indices)

        # OLD CODE

        #indices = np.vstack([np.floor(np.rad2deg(theta_sample)).astype('uint'),
        #                     np.floor(np.rad2deg(phi_sample)).astype('uint')]).T

        #self.xs = X[indices[:,0], indices[:,1]]
        #self.ys = Y[indices[:,0], indices[:,1]]


        # for interpn
        self.points = (np.arange(img_set.h), np.arange(img_set.w))
        self.xi = np.vstack([self.ys, self.xs]).T # for interpn

        super(SHProcDarioAS, self).__init__(img_set, cache_results)


    def panorama_to_surf(self, I):
        #surf_img_np = I[np.floor(self.ys).astype('uint'),
        #                np.floor(self.xs).astype('uint')]
        
        surf_img_np = interpolate.interpn(self.points, I, self.xi)
        surf_img_np = surf_img_np / 255.0 - 0.5

        return surf_img_np.tolist()


    def process_img_set(self):
        z = np.empty([self.n_bands, len(self.img_set)], dtype=float)

        for i, img in enumerate(self.img_set):
            surf_img = self.panorama_to_surf(img)
            #print surf_img
            self.hsc_wrapped.set_image(0, surf_img)
            z[:,i] = self.hsc_wrapped.calc_AS(0)

        return (z, self.n_bands, 1)


class SHProcDarioMixed(ImgProc):
    """ Dario's implementation of Spherical Harmonics. Works with panoramic images. Need to refactor and add disk shaped ones too"""


    def __init__(self, img_set, X, Y, cache_results=True, contin=3,
                n_points=2000, n_abands=120, n_bbands=3, silent=True):
        
        self.n_abands = n_abands
        self.n_bbands = n_bbands
        self.name = "shdm%sa%sb" % (n_abands, n_bbands)

        rot_par = 1             # 0:XYZ, 1:ZYZ, 2:ZYZ_FAST, 3:XYZ_FAST 4:XYZ_FREE
        resolution_deg = 6.0    # Steps for compass
        tilt_max_deg = 30.0     # Maximum tilt to check

        resolution = np.deg2rad(resolution_deg)
        tilt_max = np.deg2rad(tilt_max_deg)

        rot_dict = hsc.create_rot_par(rot_par, resolution, tilt_max)

        self.hsc_wrapped = hsc.HSC_wrapper(contin,
                                           rot_dict,
                                           n_points,
                                           n_abands,
                                           n_bbands,
                                           silent)

        (num_angles, theta_sample, phi_sample) = self.hsc_wrapped.get_angles()

        # Bring in the matlab data here for remapping nicely
        indices = np.vstack([np.floor(np.rad2deg(theta_sample)).astype('uint'),
                             np.floor(np.rad2deg(phi_sample)).astype('uint')]).T

        self.xs = X[indices[:,0], indices[:,1]]
        self.ys = Y[indices[:,0], indices[:,1]]

        super(SHProcDarioMixed, self).__init__(img_set, cache_results)


    def panorama_to_surf(self, I):
        surf_img_np = I[np.floor(self.ys).astype('uint'),
                        np.floor(self.xs).astype('uint')]
        surf_img_np = surf_img_np / 255.0 - 0.5
        return surf_img_np.tolist()


    def process_img_set(self):
        n_coeffs = self.hsc_wrapped.get_max_entries_BS() + self.n_abands
        z = np.empty([n_coeffs, len(self.img_set)], dtype=float)

        for i, img in enumerate(self.img_set):
            surf_img = self.panorama_to_surf(img)
            self.hsc_wrapped.set_image(0, surf_img)
            z[:,i] = self.hsc_wrapped.calc_mixed(0, self.n_abands, self.n_bbands)

        return (z, n_coeffs, 1)

# refactor this and above class to have one superclass
# ATTENTION THIS WAS MODDED TO USE FULL SPHERE BUT HAD MUCH BETTER RESULTS WITH CONTIN=3
class SHProcDarioBS(ImgProc):


    def __init__(self, img_set, X, Y, cache_results=True, contin=0,
                n_points=4000, n_bands=12, silent=True):

        self.n_bands = n_bands
        self.name = "shdbs%s" % n_bands

        # These are needed to create Dario's class but we should update that

        rot_par = 1             # 0:XYZ, 1:ZYZ, 2:ZYZ_FAST, 3:XYZ_FAST 4:XYZ_FREE
        resolution_deg = 6.0    # Steps for compass
        tilt_max_deg = 30.0     # Maximum tilt to check

        resolution = np.deg2rad(resolution_deg)
        tilt_max = np.deg2rad(tilt_max_deg)

        rot_dict = hsc.create_rot_par(rot_par, resolution, tilt_max)

        self.hsc_wrapped = hsc.HSC_wrapper(contin,
                                           rot_dict,
                                           n_points,
                                           n_bands,
                                           n_bands,
                                           silent)

        (self.num_angles, theta_sample, phi_sample) = self.hsc_wrapped.get_angles()

        self.theta = np.array(theta_sample)
        self.phi = np.array(phi_sample)

        # Bring in the matlab data here for remapping nicely
        indices = np.vstack([np.floor(np.rad2deg(self.theta[self.theta < np.pi/2])).astype('uint'),
                             np.floor(np.rad2deg(self.phi[self.theta < np.pi/2])).astype('uint')]).T

        self.xs = X[indices[:,0], indices[:,1]]
        self.ys = Y[indices[:,0], indices[:,1]]

        super(SHProcDarioBS, self).__init__(img_set, cache_results)


    def panorama_to_surf(self, I):
        surf_img_np = np.zeros([self.num_angles])

        surf_img_np[self.theta < np.pi/2] = I[np.floor(self.ys).astype('uint'),
                                               np.floor(self.xs).astype('uint')]
        surf_img_np = surf_img_np / 255.0 - 0.5
        return surf_img_np.tolist()
        

    def process_img_set(self):
        num_bs = self.hsc_wrapped.get_max_entries_BS()
        z = np.empty([num_bs, len(self.img_set)], dtype=float)

        for i, img in enumerate(self.img_set):
            surf_img = self.panorama_to_surf(img)
            self.hsc_wrapped.set_image(0, surf_img)
            z[:,i] = self.hsc_wrapped.calc_BS(0)

        return (z, self.n_bands, 1)

    
class GroundCHProc(ImgProc):
    """Process an image set to Grayscale with sky extraction and surrounding ground using convex hull.

    This process assumed a grayscale image.

    Attributes:
      name (str): The tag that will be used to save output.
      n_max (int): The number of Coefficients to use.
      
    """
    
    
    
    def __init__(self, img_set, cache_results=True, n_max=100):
        self.name = 'groundch'
        super(GroundCHProc, self).__init__(img_set, cache_results)
    
    
    def make_square(self, I):
        
        pixels_to_crop = (w - h) / 2
        return I[:, pixels_to_crop:w - pixels_to_crop]
    
    
    def process_img_set(self):
        print "Extracting sky and ground using convex hull"  
        
        #inverted_mask = np.invert(self.mask)
        kernel = np.ones((3, 3), np.uint8)
        
        # Todo! for now just hard code, refactor later!!
        inner_mask_radius = 88
        outer_mask_radius = 350
        
        w = self.img_set.w
        h = self.img_set.h
        centre = w / 2, h / 2
        pixels_to_crop = (w - h) / 2
        
        d_mask = donut_mask(w, h, inner_mask_radius, outer_mask_radius)
        
        selem = morp.disk(15)

        # For now we store the entire image
        z = np.empty([h * h, len(self.img_set)], dtype=float)

        # This is all VERY inefficient, but let's jsut see if it works
        for i, img in enumerate(self.img_set):
            I_seg = segment(img, d_mask)
            cv2.circle(I_seg, centre, inner_mask_radius, 255, -1)
            I_denoised = cv2.morphologyEx(I_seg, cv2.MORPH_OPEN, kernel)
            I_dilated = cv2.dilate(I_denoised, kernel, iterations = 20)
            
            contours = find_largest_contours(I_dilated)
            hull = cv2.convexHull(contours)
            I_hull = fill_contours(I_dilated.shape, hull)
            
            ground_mask = (np.logical_xor(I_seg, I_hull)).astype('uint8')
            I_ground = cv2.bitwise_and(img, img, mask=ground_mask)
            I_skyground = cv2.bitwise_or(I_seg, I_ground)

            I_skyground_cropped = I_skyground[:, pixels_to_crop:w - pixels_to_crop]
            
            # Local histogram equalisation
            I_eq_cropped = rank.equalize(I_skyground_cropped, selem=selem)
            
            # Add the ground back onto the sky
            I_hull_cropped = I_hull[:, pixels_to_crop:w - pixels_to_crop]
            I_local_eq = cv2.bitwise_and(I_hull_cropped, I_eq_cropped)
            
            z[:,i] = I_local_eq.reshape(-1)

        return(z, h, h)
    

class ZernikeProc(ImgProc):
    """Process an image set to Zernike moments.

    This process assumed a B&W sky thresholded image.

    Attributes:
      name (str): The tag that will be used to save output.
      n_max (int): The number of Coefficients to use.
      mags_only (bool): Whether to store all coefficients or just the magnitudes
      
    """
        
    def __init__(self, img_set, cache_results=True, n_max=40, mags_only=True, poly_cache_folder='data/zernpoly'):
        self.name = 'zernike'
        self.n_max = n_max
        self.mags_only = mags_only
        
        #store with different name if using magnitudes or full coefficients
        if mags_only:
            self.name = self.name + 'm'
        else:
            self.name = self.name + 'c'
        
        self.d = np.min((img_set.w, img_set.h))
        self.zernike = zernike.Zernike(d=self.d, n_max=n_max, poly_cache_folder=poly_cache_folder)
        self.n_coeffs = zernike.num_coeffs(n_max)
        super(ZernikeProc, self).__init__(img_set, cache_results)

    
    def process_img_set(self):
        """gets Zernike Coefficients for a set of images. Returns the magnitude"""
        print "Extracting Zernike coefficients"
        if self.mags_only:
            
            num_mags = zernike.num_mags(self.n_max)
            z = np.empty([num_mags, len(self.img_set)], dtype=float)

            for i, img in enumerate(self.img_set):
                if i % 1000 == 0:
                    print i
                z[:,i] = self.zernike.calc_img_mags(img)

            return (z, num_mags, 1)
        else:
            n_coeffs = zernike.num_coeffs(self.n_max)
            z = np.empty([n_coeffs, len(self.img_set)], dtype=complex)
            
            for i, img in enumerate(self.img_set):
                z[:,i] = self.zernike.calc_img_coeffs(img)
                
            return (z, n_coeffs, 1)
        #return (fd, self.num_descriptors, 1) #Not returning the magnitude gives bad results

        
        
class FDProc(ImgProc):
    """Process an image set to Fourier descriptors.

    This process assumed a B&W sky thresholded image.

    Attributes:
      name (str): The tag that will be used to save output.
      num_descriptors (int): The number of Fourier descriptors kept after truncation.
      
    """
        
    def __init__(self, img_set, cache_results=True, num_descriptors=100):
        self.name = 'fd'
        self.num_descriptors = num_descriptors
        super(FDProc, self).__init__(img_set, cache_results)
    
    
    #TODO: this doesn't include masking yet as assumes that's all been done already
    def calc_fd(self, I, num_descriptors):
        """Calculate Fourier descriptors of a binary image"""
        contours = find_largest_contours(I)
        descriptors = fd.find_descriptors(contours)
        truncated = fd.truncate_descriptor(descriptors, num_descriptors)
        return truncated
        
    
    def process_img_set(self):
        """gets FD for a set of images. Returns the magnitude"""
        print "Extracting Fourier descriptors"
        fd = np.empty([self.num_descriptors, len(self.img_set)], dtype=complex)
        
        for i, img in enumerate(self.img_set):
            fd[:,i] = self.calc_fd(img, self.num_descriptors)
            
        return (np.absolute(fd), self.num_descriptors, 1)
        #return (fd, self.num_descriptors, 1) #Not returning the magnitude gives bad results
        
        

class HuProc(ImgProc):
    """Process an image set to Hu moments.

    This process assumed a B&W sky thresholded image.

    Attributes:
      name (str): The tag that will be used to save output.
      comparison_method (int): The method of comparison to be used.
      
    """
    NO_CHANGE = 0
    CV_CONTOURS_MATCH_I1 = 1 # 1 / [sign(H) * Log(H)]
    CV_CONTOURS_MATCH_I2 = 2 # sign(H) * Log(H)
    
    num_moments = 7 # Always 7 Hu moments?
    

    def __init__(self, img_set, cache_results=True, comparison_method=1):
        self.name = 'hu'
        self.comparison_method = comparison_method
        super(HuProc, self).__init__(img_set, cache_results)    
    
    
    def process_img_set(self):
        """ Gets Hu Moments for a set of images"""
        print "Extracting Hu moments"
        
        moments = np.empty([self.num_moments, len(self.img_set)])
        
        for i, img in enumerate(self.img_set):
            moments[:,i] = cls.calc_moment(img)
            
        # Comparison methods copied from opencv.modules.imgproc.src.matchcontours.cpp
        if self.comparison_method == CV_CONTOURS_MATCH_I1:
            return 1./(np.sign(moments) * np.log10(np.abs(moments)))
        elif self.comparison_method == CV_CONTOURS_MATCH_I2:
            return np.sign(moments) * np.log10(np.abs(moments))
        else:
            return moments #Just return plain moments
    
    
    def __str__(self):
        if self.comparison_method == self.NO_CHANGE:
            return '%s_%s' % (self.img_set.name, self.name)
        else:
            return '%s_%sC%s' % (self.img_set.name, self.name, self.comparison_method)

        
        
class UnwrapProc2(ImgProc):
    """This particular implementation uses the fourier magnitude of the Fourier descriptors to get the ribbon we need"""
    
    def __init__(self, img_set, cache_results=True, num_descriptors=10, v_mid=10, w=100, scale_factor=0.01, mask=None, inner_mask=None):
        self.name = 'unwrap'
        self.num_descriptors = num_descriptors
        self.v_mid = v_mid
        self.h = v_mid * 2 + 1
        self.w = w
        self.mask = mask
        self.inner_mask = inner_mask
        self.name += str(num_descriptors)
        
        super(UnwrapProc2, self).__init__(img_set, cache_results)
        
        
    def process_img_set(self):
        print "Unwarping area around skyline"
        
        kernel = np.ones((5,5),np.uint8)
        M = np.empty([self.h * self.w, len(self.img_set)], dtype='uint8')
        
        scale_factor = 0.01        
        scales = np.linspace(1 - self.v_mid * scale_factor, 
                             1 + self.v_mid * scale_factor,
                             self.h)
        
        for i, img in enumerate(self.img_set):            
            
            # Segment the image
            I_seg = segment(img, self.mask)
            I_seg[self.inner_mask==255] = 255
            
            # Turn all sky pixels on image to 255
            I = img.copy()
            I[I_seg==255] = 255 #Uncomment to make sky white
            
            # Grow the boundary used for FD stuff
            I_seg2 = cv2.morphologyEx(I_seg, cv2.MORPH_CLOSE, kernel, iterations=10)
            I_seg2 = biggest_area(I_seg2)
            
            # Extract Fourier descriptors
            c = find_largest_contours(I_seg2)
            f = fd.find_descriptors(c)
            f_truncated = fd.truncate_descriptor(f, self.num_descriptors)
            
            # Reconstruct rough skyline
            unwrapped = np.empty([self.h, len(c)], dtype='uint8')
            
            for row_idx in range(self.h):
                f_adjusted = f_truncated.copy()
                f_adjusted[self.num_descriptors-1] *= scales[row_idx]
                c_r = fd.reconstruct(f_adjusted, len(c))
                indices = [np.clip(c_r[:,1], 0, self.img_set.h - 1),
                           np.clip(c_r[:,0], 0, self.img_set.h - 1)] # Here we are ignoring width (assuming square)
                unwrapped[row_idx, :] = I[indices]
            
            # Unwrap the ribbon around skyline
            unwrapped_shrunk = cv2.resize(unwrapped, (self.w, self.h), interpolation=cv2.INTER_AREA)            
            M[:,i] = unwrapped_shrunk.flatten()
            
        return (M, self.h, self.w)
    

    
class RoughContourProc(ImgProc):
    """Find a rough contour of the skyline using Fourier Descriptors."""
    
    def __init__(self, img_set, cache_results=True, num_descriptors=10, mask=None, inner_mask=None):
        self.name = 'roughcontour'
        self.num_descriptors = num_descriptors
        self.mask = mask
        self.inner_mask = inner_mask
        self.name += str(num_descriptors)
        
        super(RoughContourProc, self).__init__(img_set, cache_results)
        
        
    def process_img_set(self):
        print "Extracting rough skyline contours"
        kernel = np.ones((5,5),np.uint8)

        
        M = np.empty([self.img_set.h * self.img_set.w, len(self.img_set)],
                     dtype='uint8')
        
        for i, img in enumerate(self.img_set):            
            
            # Segment the image
            I_seg = segment(img, self.mask)
            I_seg[self.inner_mask==255] = 255
            I_seg = cv2.morphologyEx(I_seg, cv2.MORPH_CLOSE, kernel, iterations=10)
            I_seg = biggest_area(I_seg)
            
            
            # Turn all sky pixels on image to 255
            I = img.copy()
            #I[I_seg==255] = 255 #Uncomment to make sky white
            
            # Extract Fourier descriptors
            c = find_largest_contours(I_seg)
            f = fd.find_descriptors(c)
            f_truncated = fd.truncate_descriptor(f, self.num_descriptors)
            c_r = fd.reconstruct(f_truncated, len(c))
            
            C = cv2.drawContours(I, [c_r], -1, 255, 2)
         
            M[:,i] = C.flatten()
            
        return (M, self.img_set.h, self.img_set.w)
    
    
        
class UnwrapProc(ImgProc):
        
    def __init__(self, img_set, cache_results=True, num_descriptors=20, v_mid = 10, w=100, mask=None):
        self.name = 'unwrapsky'
        self.num_descriptors = num_descriptors
        self.v_mid = v_mid
        self.h = v_mid * 2 + 1
        self.w = w
        self.mask = mask
        self.name += str(num_descriptors)
        
        super(UnwrapProc, self).__init__(img_set, cache_results)
    
    
    def map_to_vector(self, I, contours, length):
        pixel_values = I[contours.squeeze()[:,1], contours.squeeze()[:,0]]   
        return cv2.resize(pixel_values, (1, length), interpolation=cv2.INTER_AREA).flatten()
    
    
    def unwrap_middle(self, I, c, U):
        C_filled = cv2.drawContours(np.zeros(I.shape, dtype='uint8'), [c], -1, 255, -1)
        c_list = find_contours(C_filled)
        U[self.v_mid, :] = self.map_to_vector(I, c_list[0], self.w)
        return U

    
    def unwrap_outer(self, I, c, U):
        C_dilated = cv2.drawContours(np.zeros(I.shape, dtype='uint8'), [c], -1, 1, 3)
        
        for i in range(self.v_mid):
            #c_list = find_contours(C_dilated)
            c_list = []
            _, c_list, hierarchy = cv2.findContours(C_dilated.copy(),
                                 cv2.RETR_CCOMP,  # To get a hierarchy
                                 cv2.CHAIN_APPROX_NONE,
                                 c_list)
            
            # This assumption is wrong, as somtimes inner can be more bumpy than outer?
            #c_list.sort(key=len) # First should always be the inner contour (as shorter list)
            
            # If the final value in hierarchy is -1 it's the outer, if it's 0 it's the inner
            nph = np.array(hierarchy[0])

            inner = np.argmax(nph[:,3])
            outer = np.argmin(nph[:,3])
            
            
            if len(c_list) > 1: # if there is an outer and an inner
                U[self.v_mid - (i+1), :] = self.map_to_vector(I, c_list[inner][::-1], self.w) # Reverse inner indices
                U[self.v_mid + (i+1), :] = self.map_to_vector(I, c_list[outer], self.w)
            else: # Otherwise just copy the inner
                U[self.v_mid - (i+1), :] = U[self.v_mid - i, :] # Reverse inner indices
                U[self.v_mid + (i+1), :] = self.map_to_vector(I, c_list[outer], self.w)

            C_dilated = cv2.drawContours(C_dilated, c_list, -1, 1, 3) #Add the contours to the outline.
         
        return U
    
    
    def unwrap(self, I, c):
        U = np.zeros((self.h, self.w))
        U = self.unwrap_middle(I, c, U)
        U = self.unwrap_outer(I, c, U)
        return U
    
    
    def process_img_set(self):
        print "Unwarping area around skyline"
        
        U = np.empty([self.h * self.w, len(self.img_set)])
        
        for i, img in enumerate(self.img_set):            
            
            # Segment the image
            I_seg = segment(img, self.mask)
            
            # Extract Fourier descriptors
            c = find_largest_contours(I_seg)
            f = fd.find_descriptors(c)
            f_truncated = fd.truncate_descriptor(f, self.num_descriptors)
            
            # Reconstruct rough skyline
            c_r = fd.reconstruct(f_truncated, len(c))
            
            # Unwrap the ribbon around skyline
            unwrapped = self.unwrap(img, c_r)
            U[:,i] = unwrapped.flatten()
        
        return (U, self.h, self.w)
        
        
        