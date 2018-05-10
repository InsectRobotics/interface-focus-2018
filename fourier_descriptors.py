# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 13:03:59 2015

@author: tomish
"""

import numpy as np

def find_descriptors(contours, remove_dc=False):
    """Find the Fourier Descriptors for an image"""
    contour_complex = np.empty(contours.shape[0], dtype=complex)
    contour_complex.real = contours[:, 0]
    contour_complex.imag = contours[:, 1]

    #We can optionally remove the DC component (perhaps not the right place here)
    if remove_dc:
        contour_complex -= np.mean(contour_complex) 

    # Do we need to re-sample here so we can compare shapes
    # with different sized contours?

    descriptors = np.fft.fft(contour_complex)
    return descriptors


def truncate_descriptor(descriptors, degree):
    """this function truncates an unshifted fourier descriptor array
    and returns one also unshifted"""
    
    descriptors = np.fft.fftshift(descriptors)
    center_index = len(descriptors) / 2
    descriptors = descriptors[
        center_index - degree / 2:center_index + degree / 2]
    descriptors = np.fft.ifftshift(descriptors)
    return descriptors


def normalise_descriptors(descriptors):
    # Scale
    # descriptors / np.linalg.norm(descriptors) # or use first descriptor??

    # Translation

    # Rotation (phase)
    pass


def pad_descriptors(descriptors, length):
    """Adds zeros to an unshifted fourier descriptor array
    and returns one also unshifted"""
    
    padded = np.zeros(length, dtype='complex')
    degree = len(descriptors)
    descriptors = np.fft.fftshift(descriptors)
    
    center_index = length / 2
    left_index = center_index - degree / 2 # Left index always round down
    right_index = int(round(center_index + degree / 2.0)) # Right index rounded up 
           
    padded[left_index:right_index] = descriptors
    padded = np.fft.ifftshift(padded)
    return padded


def reconstruct(descriptors, length):
    """Reconstructs a list of contour coordinates from the fourier descriptors
    Takes the length of the original contours to know how much to pad."""
    
    padded = pad_descriptors(descriptors, length) # Pad descriptors
    inversed = np.fft.ifft(padded) # Inverse Fourier transform
    reconstructed = np.rint(np.column_stack((inversed.real, inversed.imag))).astype('int') # Convert to coordinates
    
    return reconstructed
