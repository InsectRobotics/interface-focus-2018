import os.path
import numpy as np
import numpy.ma as ma # Masked array
from scipy.misc import factorial as fac


def zernike_rad(m, n, rho):
    """
    Calculate the radial component of Zernike polynomial (m, n) 
    given a grid of radial coordinates rho.
    """
    
    #Todo: This could be sped up A LOT by caching all this factorial stuff
    
    if (n < 0 or m < 0 or abs(m) > n):
        raise ValueError
    
    if ((n - m) % 2):
        return rho*0.0
    
    pre_fac = lambda k: (-1.0) ** k * fac(n - k) / (fac(k) * fac((n + m) / 2.0 - k) * fac((n - m) / 2.0 - k))
    
    return np.sum(pre_fac(k) * rho**(n - 2.0 * k) for k in xrange((n - m) / 2 + 1))


def zernike(m, n, rho, phi):
    """
    Calculate Zernike polynomial (m, n) given a grid of radial
    coordinates rho and azimuthal coordinates phi.
    """
    
    if (m > 0): return zernike_rad(m, n, rho) * np.cos(m * phi)
    #if (m < 0): return zernike_rad(-m, n, rho) * np.sin(-m * phi)
    if (m < 0): return zernike_rad(-m, n, rho) * np.sin(m * phi)
    return zernike_rad(0, n, rho)


def zernikel(j, rho, phi):
    """
    Calculate Zernike polynomial with Noll coordinate j given a grid of radial
    coordinates rho and azimuthal coordinates phi.
    """
    
    (n, m) = noll_to_nm(j)
    return zernike(m, n, rho, phi)


def noll_to_nm(j):
    """
    Convert from noll index j to n, m
    
    This is NOT actually the noll index, as can be seen from https://oeis.org/A176988
    but rather the natural arrangement
    """
    n = 0
    while (j > n):
        n += 1
        j -= n

    m = -n + 2 * j
    return (n, m)


def nm_to_noll(n, m):
    """
    Inverse of noll_to_nm. As with that function not actually noll index, but works for now.
    """
    j = num_coeffs(n-1)
    m_list = get_m_indices(n)
    j += np.where(m_list == m)[0][0]
    return j


def get_m_indices(n):
    """
    m indices for matching n in order of natural (triangle) arrangement.
    """
    m = np.arange(-n, n+1, 2)
    return m
    

def num_coeffs(n_max):
    """The number of coefficients needed for a certain maximum n."""
    return np.sum(np.arange(n_max + 2))


def num_mags(n_max):
    n_mags =  2 * np.sum(np.arange((n_max + 4)/2))
    if n_max % 2 == 0:
        n_mags -= ((n_max/2) + 1)
    return n_mags


def cov_mat_inv(Z):
    """Returns inverse cov matrix, used to set magnitude of all polynomials the same. (I think)"""
    # Both of these give slightly different results... why?!
    #cov_mat = np.linalg.pinv(np.array([[np.sum(Z_i * Z_j) for Z_i in zern_list] for Z_j in zern_list]))
    return np.linalg.pinv(np.dot(Z, Z.T))


def reconstruct(coeffs, zern_list):
    """ Reconstruct the image using square polynomials and coefficients. """
    return sum(c * Z for (c, Z) in zip(coeffs, zern_list))


def get_magnitudes(c):
    magnitude = []

    for j, coeff in enumerate(c):
        n, m = noll_to_nm(j)
        if m == 0:
            magnitude.append(c[j])
        elif m < 0:
            j_match = nm_to_noll(n, -m)
            magnitude.append(np.linalg.norm(np.array([c[j], c[j_match]])))

    return np.array(magnitude)


class Zernike(object):
    """
    A class to repeatedly access some basic radial polynomials at a given resolution.
    """
        
    # I don't think we need height and width as always the same
    def __init__(self, d, n_max, poly_cache_folder="data/zernpoly"):
        """
        Keyword arguments:
        d -- the diameter of the zernike function (d = w = h)
        n_max -- the numbers of harmonics?
        """
        self.d = d
        self.poly_cache_folder = poly_cache_folder
        
        x_range = np.linspace (-1, 1, d)
        xx, yy = np.meshgrid(x_range, x_range)
        grid_rho = (xx ** 2 + yy ** 2) ** 0.5
        grid_phi = np.arctan2(xx, yy)
        self.grid_mask = grid_rho <= 1
        self.rho = np.reshape(grid_rho[self.grid_mask], -1)
        self.phi = np.reshape(grid_phi[self.grid_mask], -1)
        
        coeff_count = num_coeffs(n_max)
        self.pixels_per_poly = np.sum(self.grid_mask)
        
        self.Z = np.zeros([coeff_count, self.pixels_per_poly])
            
        self.m = np.zeros(coeff_count)
        self.n = np.zeros(coeff_count)

        for j in range(coeff_count):
            self.n[j], self.m[j] = noll_to_nm(j)
            
        self.epsilon = np.ones(coeff_count)
        self.epsilon[self.m == 0] = 2.0


        # Just a hacky way of caching previously calculated polynomials
        j = 0
        for n in range(n_max + 1):
            m_list = get_m_indices(n)
            m_count = len(m_list)
            
            cache_file = self.get_cache_file_name(n) # Try load from backup
            if os.path.isfile(cache_file):
                data = np.load(cache_file)
                self.Z[j:j+m_count, :] = data['N']
            else:
                N = self.zernike_by_rad_deg(n, m_list)
                self.Z[j:j+m_count, :] = N
                print "saving to", cache_file
                np.savez(cache_file, N=N)
            
            j += m_count

        # I'M NOT SURE THIS IS CORRECT
        # First option leads to very large coeffs!
        #self.zern_cov_inv = np.linalg.pinv(np.cov(self.Z))
        self.zern_cov_inv = cov_mat_inv(self.Z)
    
    
    def _project_to_square(self, vals):
        """Reprojects the values back to a square image"""
        X = np.zeros([self.d ** 2])
        X[np.reshape(self.grid_mask, -1)] = vals
        return np.reshape(X, (self.d, self.d))
    
    
    def zernike_by_rad_deg(self, n, m_list):
        """Return Zernike polynomials by radial degree n"""
        X = np.zeros([len(m_list), self.pixels_per_poly])
        
        for i, m in enumerate(m_list):
            z = zernike(m, n, self.rho, self.phi)
            X[i, :] = z
        
        return X

        
    def reconstruct(self, c):
        """Reconstruct an image using coefficients"""
        j = c.size
        x = np.dot(self.Z[:j,:].T, c)
        I = self._project_to_square(x)
        return ma.masked_array(I, mask=np.logical_not(self.grid_mask))
    
    
    def reconstruct_binary(self, c, threshold=None):
        """Create a binary image using input of coefficients"""
        if threshold is None:
            threshold = c[0]
        I = self.reconstruct(c)
        return I > threshold
    
        
    def calc_coeff(self, x, j):
        """Calculate a single coefficient"""
        return np.dot(self.zern_cov_inv[j, :], np.dot(self.Z[j, :], x))
    
    
    def calc_coeffs(self, x, j_max=None):
        """Calculate the coefficients of vectorised masked image x"""
        #if j_max is None:
        #    j_max = self.Z.shape[0]
        #return np.dot(self.zern_cov_inv[0:j_max, 0:j_max], np.dot(self.Z[0:j_max, :], x))

        if j_max is None:
            j_max = self.Z.shape[0]
        
        return (2 * self.n[0:j_max] + 2) / (self.epsilon[0:j_max] * np.pi) * np.dot(self.Z[0:j_max, :], x)
    
    
    def calc_img_coeffs(self, I, j_max=None):
        """Calculate coefficients of an image passed in"""
        return self.calc_coeffs(I[self.grid_mask].reshape(-1), j_max)
    
    
    def calc_img_mags(self, I, j_max=None):
        """Return magnitudes of image (rotation invariant)"""
        mags = get_magnitudes(self.calc_img_coeffs(I, j_max))
        return mags
    
                              
    def view_polynomial(self, j):
        """Visualise a single polynomial"""
        P = self._project_to_square(self.Z[j,:])
        return ma.masked_array(P, mask=np.logical_not(self.grid_mask))
    
    
    def get_cache_file_name(self, n):
        """The filename to store all radial basis functions with certain size and n"""
        return '%s/zernike%s_%s.npz' % (self.poly_cache_folder, self.d, n)
    