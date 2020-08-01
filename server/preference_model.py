import numpy as np


class SphericalLinearPreferenceModel:
    def __init__(self, shape=512, rng=None):
        """
        Create model object.
        :param shape: shape of the vector to learn the preference from.
        """
        self._rng = rng or np.random.RandomState(0)
        self._shape = shape
        # Learnable parameters
        self._training_examples = []

    @property
    def is_random(self):
        return len(self._training_examples) == 0

    def train(self, training_examples):
        self._training_examples = training_examples
        # self._r0 = -5

    def generate(self, size, variance=0):
        """
        Generate new data for current model parameters.
        :param size the number of vectors to generate.
        :param variance the larger the factor, the more mutation has the output
        :return: an array of vectors similar to those used for training.
        """
        if self.is_random:
            return sample_uniform_on_sphere(self._rng, self._shape, size)

        # For variance from 0 to 4
        # a, scale, std
        params = [
            (10, 1, .02),
            (3, 1, .03),
            (1, 1, .05),     # Middle range - a uniform dist. in convex combination of training examples
            (1, 1.5, .07),   # Start going outside of the convex combination of training examples
            (1.2, 2.0, .1)  # Concentrate in the middle, as the values at the boarder have little visual difference
        ][variance]

        k = len(self._training_examples)

        # Convert to spherical coordinates
        _, phi = cartesian_to_spherical_n(self._training_examples)

        # [0, pi] -> [-pi, pi] for all but the last
        phi[:, :-1] = 2 * phi[:, :-1] - np.pi

        if k == 1:
            # Only one training example, it will be varied later by a normal "noise".
            output_phi = phi
        else:
            # Mix k training examples
            if hasattr(self, '_r0') and k == 2:
                # Go through convex combinations.
                self._r0 += 0.1
                r = np.array([self._r0, 1 - self._r0]).reshape(size, k)
            elif k == 2 and False:
                # Simple test for n = 2
                r = self._rng.standard_normal(size=1) * variance + 0.5
                r = np.array([r, 1-r]).reshape(size, k)
            else:
                # Random coefficients of shape (size, k)
                r = scaled_dirichlet(self._rng, k=k, size=size, a=params[0], scale=params[1])

            print(r, r.sum(axis=1))

            # Sines and cosines of shape (size, k, 511)
            sin = np.broadcast_to(np.sin(phi), (size,) + phi.shape)
            cos = np.broadcast_to(np.cos(phi), (size,) + phi.shape)

            # Expand to shape (size, k, 1)
            r = np.expand_dims(r, 2)

            sin = sin * r
            cos = cos * r

            # Linear combinations of shape (size, 511)
            output_phi = np.arctan2(sin.sum(axis=1), cos.sum(axis=1))

        # Add normal "noise"
        output_phi += self._rng.normal(scale=params[2], size=output_phi.shape)

        # [-pi, pi] -> [0, pi] for all but the last
        output_phi[:, :-1] = (output_phi[:, :-1] + np.pi) / 2

        output = spherical_to_cartesian_n(np.ones((size, 1)), output_phi)

        return output


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

def sample_uniform_on_sphere(rng, dim, size):
    """
    Sample uniform random points on n-sphere.
    See
    https://mathworld.wolfram.com/HyperspherePointPicking.html
    https://stackoverflow.com/questions/15880367/python-uniform-distribution-of-points-on-4-dimensional-sphere
    http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/

    :param rng a random number generator to use.
    :param dim: dimension of the space (e.g. 3 for 3d).
    :param size: number of points.
    :return: an array uniformly distributed points. The normalization to the unit length is not done to avoid
    division by zero and because the is done by the model.
    """
    return rng.normal(size=(size, dim))


def scaled_dirichlet(rng, k, a, size=None, scale=1):
    """
    Sample from a symmetric Dirichlet distribution of dimension k and parameter a, scaled around its mean.

    It generates vectors of dimension k. Sum of the elements of the vectors is 1.
    The elements are in the range [(1-scale) / k, (scale * (k-1) + 1) / k]
    The mean of each element is 1 / k.
    The variance is scale**2 * (k-1) / k**2 / (k * a + 1).

    :param rng: random number generator.
    :param k: dimensionality.
    :param a: distribution parameter in [eps, +inf]. eps shall be > 0.01 or so to avoid nans.
    :param size: output shape, the output will have a shape (size, k). If size is None, a vector of size k is returned.
    :param scale: scale factor.
    :return: a size vectors with k elements each.
    """
    if type(size) == int:
        size = (size,)
    if False:
        # Use the native numpy function.
        x =rng.dirichlet(np.full(k, a), size)
    else:
        # Use the gamma distribution as in tensorflow.js.
        y = rng.gamma(np.full(k, a), size=size + (k,))
        x = y / y.sum(axis=-1, keepdims=True)

    mean = 1. / k
    return (x - mean) * scale + mean


def normalize_angle(angle, period=np.pi * 2, start=None):
    """
    Transforms the angle to the value in range [start, start + period].

    :param angle: angle
    :param period: period
    :param start: minimal value of the resulting angle. If None, is set to -period/2.
    :return: converted angle
    """

    if start is None:
        start = -period / 2

    return (angle - start) % period + start