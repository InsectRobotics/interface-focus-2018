# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 12:31:47 2015

@author: tomish
"""
import os.path
import re
import glob
from PIL import Image
import cv2
import numpy as np

def convert_to_gs_enhance_blue(I):
    # added small constant to avoid division by zero problems
    I2 = 3 * np.square(I[:,:,0].astype('float')) / (np.sum(I.astype('float'), axis=2) + 0.0001)
    if np.max(I2) > 255:
        I2 = I2 / np.max(I2) * 255
    return I2.astype('uint')
    #return np.clip(3 * np.square(I[:,:,0].astype('float')) / (np.sum(I.astype('float'), axis=2)+ 0.0001), 0, 255).astype('uint')

class ImgSet(object):
    """Stores a collection of images.
    
    Each image is stored as a column vector. Currently we only support monochromatic images.
    
    Attributes:
      data_folder (str):
      name (str):
      file_type (str):
      files (list of str):
      M (numpy array):
      w (int): Width of each image
      h (int): height of each image
      
    """
    
    #data_folder = ''
    #name = '' # Change this to batch_name to avoid confusion with setter()
    #file_type = ''
    #files = []
    #M = None
    #w = None
    #h = None
    
    def __init__(self, name, M=None, w=None, h=None, first=None, last=None, step=1, data_folder='data', file_type='png', force_cache=False, enhance_blue=False):
        self.data_folder = data_folder
        
        self.file_type = file_type
        self.enhance_blue = enhance_blue

        # If we passed the image matrix in just use that
        if M is not None and w and h:
            print "loading from matrix"
            self.name = name
            self.M = M
            self.w = w
            self.h = h
        else: # Otherwise load from files or cache
            
            if not first or not last:
                self.name = name
            else:
                self.name = '%s_%s-%s' % (name, first, last)

            # Try loading from cache
            cache_file = self.get_cache_file()
            if cache_file:
                print "Loading cache file: %s" % cache_file
                data = np.load(cache_file)
                self.M = data['M']
                self.w = data['w']
                self.h = data['h']
            # Load from files
            else:
                print "loading from image files in %s" % self.get_directory()
                files = self.get_file_names()
                
                if not first or not last:
                    first = 0
                    last = len(files)
                
                # Todo: add Try statement and warning: make sure you have correct filetype bla
                self.files = files[first:last:step]
                img = Image.open(self.files[0]) # Lazy so won't load into memory
                self.w, self.h = img.size
                
                if force_cache:
                    M = np.empty([self.w * self.h, len(self)], dtype='uint8')
                    for i, img in enumerate(self):
                        M[:, i] = img.reshape(-1)
                    self.M = M
                    self.save()
    
    
    @classmethod
    def load_from_cache(cls, cache_file):
        """Factory to create an ImgSet from a cache file"""
        if os.path.isfile(cache_file):
            data = np.load(cache_file)
            return cls(name=data['name'], M=data['M'], w=data['w'], h=data['h'])
        else:
            # Throw error here
            return None
    
    
    def get_cache_file_name(self):
        """Return the corresponding cache file name"""
        return os.path.join(self.data_folder, self.name + '.npz')
        

    def get_cache_file(self):
        """Returns corresponding cache file name if it exists in filesystem"""
        cache_file = self.get_cache_file_name()
        if os.path.isfile(cache_file):
            return cache_file
        else:
            # Throw error here
            return None


    def get_directory(self):
        """Strips indices from name to give directory of files"""
        directory_name = re.split('_[0-9]+\-[0-9]+', self.name)[0]
        return os.path.join(self.data_folder, directory_name)
        
    
    def get_file_names(self):
        """Returns sorted filenames in image set directory"""
        directory = self.get_directory()
        return sorted(glob.glob('%s/*.%s' % (directory, self.file_type)))
        

    def save(self):
        """Save the image set to compressed Numpy format"""
        cache_file = self.get_cache_file_name()
        np.savez_compressed(cache_file, name=self.name, M=self.M, w=self.w, h=self.h)


    def images_from_files(self):
        """Iterates through files to yield images"""
        for file_name in self.files:
            
            if self.enhance_blue:
                I = cv2.imread(file_name, 1)
                img = convert_to_gs_enhance_blue(I)
            else:
                img = cv2.imread(file_name, 0) # Grayscale

            yield img
    
    
    def images_from_matrix(self):
        """Iterates through matrix to yield images"""
        # convert boolean to 8 bit
        if self.M.dtype == np.dtype('bool'):
            M_grayscale = self.M.T.astype('u1') * 255
        # Convert positive and negative changes to 8 bit
        elif np.min(self.M) == -1 and np.max(self.M) == 1:
            M_grayscale = ((self.M.T + 1) * 255 / 2).astype('u1')
        else:
            M_grayscale = self.M.T
        
        for row in M_grayscale:
            yield row.reshape(self.h, self.w)

    
    def __iter__(self):
        """Iterates through images"""
        if hasattr(self, 'M') and self.M is not None:
            return self.images_from_matrix()
        else:
            return self.images_from_files()          


    def __len__(self):
        """Number of images in the image set"""
        if hasattr(self, 'M') and self.M is not None:
            return self.M.shape[1]
        else:
            return len(self.files)
        
    
    def __getitem__(self, key):
        """Load an image by index"""
        if hasattr(self, 'M') and self.M is not None:
            if self.M.dtype == np.dtype('bool'):
                row = self.M[:,key].astype('u1') * 255
            else:
                row = self.M[:,key]
            return row.reshape(self.h, self.w)
        else:
            print "loading %s" % self.files[key]
            return cv2.imread(self.files[key], 0)            

    
    def view_as_movie(self):
        """View the image set as movie"""
        # TODO: allow this to be done from a particular index.
        for img in self:
            cv2.imshow('frame', img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        
    
    def view_frame(self, idx):
        """View a single frame by index"""
        I = self[idx]
        cv2.imshow('frame', I)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
