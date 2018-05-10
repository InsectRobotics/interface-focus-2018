# Some SH functions will be housed here

import numpy as np
import numpy.ma as ma # Masked array
import pyshtools as shtools
from sklearn.neighbors import NearestNeighbors


def generate_spherical_coords(w):
    x_range = np.linspace (-1, 1, w)
    xx, yy = np.meshgrid(x_range, x_range)

    phi = np.pi / 2 - np.arccos((xx ** 2 + yy ** 2) ** 0.5)
    theta = np.arctan2(xx, yy) + np.pi

    mask = np.isnan(phi) # Create a mask to block any pixels outside the sphere
    theta_masked = ma.masked_array(theta, mask=mask)
    phi_masked = ma.masked_array(phi, mask=mask)

    return (theta_masked, phi_masked, mask)


def create_mapping(theta_masked, phi_masked, theta_grid, phi_grid):
        train = np.array(zip(theta_masked.compressed(), phi_masked.compressed()))
        
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(train)
        
        test = np.array(zip(theta_grid.reshape(-1), phi_grid.reshape(-1)))
        (dist, mapping) = neigh.kneighbors(test)
        return mapping
    

def convert_to_grid(I):
    """Currently using nearest neighbour approach to map onto a grid. There must be a much nicer way!"""
    w_img = I.shape[0]
    w_r = w_img * 2

    theta_masked, phi_masked, mask = generate_spherical_coords(w_img)
    
    phi_grid, theta_grid = np.mgrid[0:np.pi:w_r * 1j, 0:2 * np.pi:w_r * 1j]
    
    mapping = create_mapping(theta_masked, phi_masked, theta_grid, phi_grid)
    
    inverted_mask = np.invert(mask)
    img_vector = I[inverted_mask].reshape(-1)
    values = img_vector[mapping]
    return values.reshape(w_r, w_r)


def calc_coeffs(img_grid):
    coeffs = shtools.SHExpandDH(img_grid) # Real coefficients
    return coeffs


def truncate_coeffs(C, n):
    C_truncated = C.copy()
    mask = np.zeros(C.shape, dtype='bool')
    mask[:, 0:n, 0:n] = 1
    C_truncated[np.invert(mask)] = 0
    return C_truncated


def reconstruct_grid(C):
    return shtools.MakeGridDH(C, csphase=-1)