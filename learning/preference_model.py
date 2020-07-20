import numpy as np


class PreferenceModel:
    def __init__(self, dna_shape, default_std=0.01):
        super().__init__(dna_shape)
        self._default_std = default_std
        # Init mean, std
        self.train([])

    @staticmethod
    def _mean_of_angles(angles, weights):
        s = np.sin(angles)
        c = np.cos(angles)
        if weights is not None:
            weights = weights.reshape(-1, 1)
            s *= weights
            c *= weights
        m = np.arctan2(s.sum(axis=0), c.sum(axis=0))
        return m

    def train(self, elite_dna, weights=None):
        if len(elite_dna) == 0:
            # Reset mean, std
            self._mean = None
            self._std = None
        else:
            # Update mean, std.
            assert (elite_dna.ndim == 2)
            r, phi = cartesian_to_spherical_n(elite_dna)
            self._mean = self._mean_of_angles(phi, weights)
            if len(elite_dna) > 1:
                phi_c = phi - self._mean
                a = np.abs(phi_c)
                self._std = self._mean_of_angles(a, weights)
            else:
                self._std = np.full(self._mean.shape, self._default_std, dtype=np.float32)

    def generate(self, count):
        """
        Generate new DNAs with current model parameters.
        :return: a vector of DNAs.
        """
        if self._std is None:
            # Generate uniform distribution on the sphere
            mean = np.zeros(self._dna_shape, dtype=np.float32)
            std = np.full(self._dna_shape, 1, dtype=np.float32)
            cov = np.diag(np.array(std ** 2))
            dna = self._rng.multivariate_normal(mean=mean, cov=cov, size=count)
        else:
            std = self._std * self._std_scale
            cov = np.diag(np.array(std ** 2) + 1e-5)
            phi = self._rng.multivariate_normal(mean=self._mean, cov=cov, size=count, check_valid="ignore").astype(
                np.float32)
            # TODO(ia): do we have to make sure that phi is in required range?
            dna = spherical_to_cartesian_n(np.full((len(phi), 1), 1), phi)
        return dna.astype(dtype=np.float32)


def spherical_to_cartesian_n(r, phi):
    """
    Convert spherical to cartesian coordinates in n dimensions.

    See https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates.
    :param r: radius (a scalar or a column of scalars).
    :param phi: a vector of anlges of size n-1 or column of such vectors.
    phi[0:n-2] vary in range [0, pi], phi[n-1] in [0, 2*pi] or in [-pi, pi].
    :return: cartesian coordinates (a vector of size n or a column of such vectors).
    """
    ones_shape = (1,) if phi.ndim == 1 else phi.shape[:1] + (1,)
    ones = np.full(ones_shape, 1.0, dtype=phi.dtype)
    sinphi = np.sin(phi)
    axis = 0 if phi.ndim == 1 else 1
    sinphi = np.cumprod(sinphi, axis=axis)
    sinphi = np.concatenate((ones, sinphi), axis=axis)
    cosphi = np.cos(phi)
    cosphi = np.concatenate((cosphi, ones), axis=axis)

    x = sinphi * cosphi * r

    return x


def cartesian_to_spherical_n(x, eps=1e-10):
    """
    Converts cartesian to spherical coordinates in n dimensions.

    See https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates.
    :param x: cartesian coordinates (a vector or an array of row vectors).
    :param eps: elements of x < eps are considered to be 0.
    :return: r, phi
    r: radius (a scalar or a column of scalars)
    phi: a vector of angles of size n-1 or column of such vectors.
    phi[0:n-2] vary in range [0, pi], phi[n-1] in [-pi, pi].
    """
    is_reshaped = False
    if x.ndim == 1:
        is_reshaped = True
        x = x.reshape(1, -1)

    x2 = np.flip(x * x, axis=1)
    n = np.sqrt(np.cumsum(x2, axis=1))

    n = np.flip(n, axis=1)
    r = n[:, 0].reshape(-1, 1)
    n = n[:, :-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        xn = x[:, :-1] / n

    phi = np.arccos(xn)

    phi[n < eps] = 0

    #
    # The description in wikipedia boils down to changing the sign of the  phi_(n-1) (using 1-based indexing)
    # if and only if
    # 1. there is no k such that x_k != 0 and all x_i == 0 for i > k
    # and
    # 2. x_n < 0

    s = x[:, -1] < 0
    phi[s, -1] *= -1

    if is_reshaped:
        r = r.item()
        phi = phi.reshape(phi.size)

    return r, phi


def mean_of_angles(angles, axis=None):
    """
    Compute mean of angular values as described in https://en.wikipedia.org/wiki/Mean_of_circular_quantities.

    :param angles: an array of angles.
    :param axis: Axis or axes along which the means are computed.
    :return: mean.
    """
    s = np.sin(angles)
    c = np.cos(angles)
    m = np.arctan2(s.sum(axis=axis), c.sum(axis=axis))
    return m