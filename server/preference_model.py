import numpy as np

def get_variance_factor(value):
    """
    Get a floting-point variance factor for a given intuitive value.
    :param value: a number >= 0, 0 - no or very little changes, 8 - very strong changes.
    :return: a factor.
    """
    return np.power(2.0, value - 4.0)

class SphericalCoordinatesPreferenceModel:
    def __init__(self, shape=512, default_std=0.01, rng=None):
        """
        Create model object.
        :param shape: shape of the vector to learn the preference from.
        :param default_std: the value of standard deviation to use if we learn from only one training example.
        """
        self._rng = rng or np.random.RandomState(0)
        self._shape = shape
        self._default_std = default_std
        # Learnable parameters
        self._mean = None
        self._std = None

    @property
    def is_random(self):
        return self._std is None

    def train(self, training_examples):
        if len(training_examples) == 0:
            # Reset the trainable parameters and use a uniform distribution.
            self._mean = None
            self._std = None
        else:
            # Learn mean, std.
            assert (training_examples.ndim == 2)
            r, phi = cartesian_to_spherical_n(training_examples)
            self._mean = mean_of_angles(phi, axis=0)
            if len(training_examples) > 1:
                phi_c = phi - self._mean
                a = np.abs(phi_c)
                self._std = mean_of_angles(a, axis=0)
            else:
                self._std = np.full(self._mean.shape, self._default_std, dtype=np.float32)

    def generate(self, size, mutation_factor=1):
        """
        Generate new data for current model parameters.
        :param size the number of vectors to generate.
        :param mutation_factor the larger the factor, the more mutation has the output
        :return: an array of vectors similar to those used for training.
        """
        mutation_factor = get_variance_factor(mutation_factor)
        if self.is_random:
            output = sample_uniform_on_sphere(self._rng, self._shape, size)
        else:
            std = self._std * mutation_factor
            cov = np.diag(np.array(std ** 2) + 1e-5)
            phi = self._rng.multivariate_normal(mean=self._mean, cov=cov, size=size, check_valid="ignore").astype(
                np.float32)
            # TODO(ia): do we have to make sure that phi is in required range?
            output = spherical_to_cartesian_n(np.full((len(phi), 1), 1), phi)
        return output.astype(dtype=np.float32)


class SphericalCoordinates2PreferenceModel:
    def __init__(self, shape=512, default_std=0.01, rng=None):
        """
        Create model object.
        :param shape: shape of the vector to learn the preference from.
        :param default_std: the value of standard deviation to use if we learn from only one training example.
        """
        self._rng = rng or np.random.RandomState(0)
        self._shape = shape
        self._default_std = default_std
        # Learnable parameters
        self._training_examples = []

    @property
    def is_random(self):
        return len(self._training_examples) == 0

    def train(self, training_examples):
        self._training_examples = training_examples

    def generate(self, size, mutation_factor=0):
        """
        Generate new data for current model parameters.
        :param size the number of vectors to generate.
        :param mutation_factor the larger the factor, the more mutation has the output
        :return: an array of vectors similar to those used for training.
        """
        mutation_factor = get_variance_factor(mutation_factor)
        if self.is_random:
            output = sample_uniform_on_sphere(self._rng, self._shape, size)
        elif len(self._training_examples) == 1:
            output = np.tile(self._training_examples, (size, 1))
        else:
            assert size == 1
            assert len(self._training_examples) == 2, "This is a simple test for 2 training examples: tc0 * a + tc1 * (1-a)"

            r, phi = cartesian_to_spherical_n(self._training_examples)

            # [0, pi] -> [-pi, pi] for all but the last
            phi[:, :-1] = 2 * phi[:, :-1] - np.pi

            a = self._rng.standard_normal(size=1) * mutation_factor + 0.5
            print(a)
            a = np.array([a, 1-a]).reshape(-1, 1)
            s = np.sin(phi) * a
            c = np.cos(phi) * a
            output_phi = np.arctan2(s.sum(axis=0, keepdims=True), c.sum(axis=0, keepdims=True))
            # [-pi, pi] -> [0, pi] for all but the last
            output_phi[:, :-1] = (output_phi[:, :-1] + np.pi) / 2

            output = spherical_to_cartesian_n(np.ones((size, 1)), output_phi)

        return output


class LinearPreferenceModel:
    def __init__(self, shape=512, default_cov=0.01, rng=None):
        """
        Create model object.
        :param shape: shape of the vector to learn the preference from.
        :param default_std: the value of standard deviation to use if we learn from only one training example.
        """
        self._rng = rng or np.random.RandomState(0)
        self._shape = shape
        self._default_cov = default_cov
        # Learnable parameters are inside the DimensionalityReduciton instance.
        self._training_examples = []

    @property
    def is_random(self):
        return len(self._training_examples) == 0

    def train(self, training_examples):
        # Todo(ia): division by zero?
        self._training_examples = training_examples / np.linalg.norm(training_examples, axis=1, keepdims=True)

    def generate(self, size, mutation_factor=1):
        """
        Generate new data for current model parameters.
        :param size the number of vectors to generate.
        :param mutation_factor the larger the factor, the more mutation has the output
        :return: an array of vectors similar to those used for training.
        """
        mutation_factor = get_variance_factor(mutation_factor)
        if self.is_random:
            output = sample_uniform_on_sphere(self._rng, self._shape, size)
        elif len(self._training_examples) == 1:
            output = np.tile(self._training_examples, (size, 1))
        else:
            # TODO(ia): support size > 1
            assert size == 1
            mean = self._training_examples.mean(axis=0)
            tc = self._training_examples - mean
            assert len(tc) == 2, "This is a simple test for 2 training examples: tc0 * a + tc1 * (1-a)"
            # TODO(ia): now there is a problem if a > 2 or a < -1, it does not really change the pictures visually
            # and therefore it is unclear how to implement large mutation factors.
            a = self._rng.standard_normal(size=1) + 0.5
            print(a)
            output = tc[0] * a * mutation_factor + tc[1] * (1-a) * mutation_factor
            output += mean
            output = output.reshape(1, self._shape)
        return output


class DimRedPreferenceModel:
    def __init__(self, shape=512, default_cov=0.01, rng=None):
        """
        Create model object.
        :param shape: shape of the vector to learn the preference from.
        :param default_std: the value of standard deviation to use if we learn from only one training example.
        """
        self._rng = rng or np.random.RandomState(0)
        self._shape = shape
        self._default_cov = default_cov
        # Learnable parameters are inside the DimensionalityReduciton instance.
        self._dim_red = None

    @property
    def is_random(self):
        return self._dim_red is None

    def train(self, training_examples):
        if len(training_examples) == 0:
            # Reset the trainable parameters and use a uniform distribution.
            self._dim_red = None
        else:
            # Put the points onto the sphere.
            # Todo(ia): division by zero?
            training_examples /= np.linalg.norm(training_examples, axis=1, keepdims=True)
            self._dim_red = DimensionalityReduction(training_examples, 0.99)

    def generate(self, size, mutation_factor=1):
        """
        Generate new data for current model parameters.
        :param size the number of vectors to generate.
        :param mutation_factor the larger the factor, the more mutation has the output
        :return: an array of vectors similar to those used for training.
        """
        mutation_factor = get_variance_factor(mutation_factor)
        if self.is_random:
            output = sample_uniform_on_sphere(self._rng, self._shape, size)
        else:
            cov = np.maximum(self._dim_red.cov, self._default_cov) * mutation_factor
            output_r = self._rng.multivariate_normal(mean=np.zeros_like(cov), cov=np.diag(cov), size=size)
            print(output_r)
            output = self._dim_red.restore_dim(output_r)
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


def sample_von_moses_fisher(rng, dim, mu, kappa, size, epsilon=1e-5):
    """
    Sample from von Moses-Fisher distribution n-sphere.

    We use the rejection sampling algorithm: https://en.wikipedia.org/wiki/Rejection_sampling

    :param rng a random number generator to use.
    :param dim: dimension of the space (e.g. 3 for 3d).
    :param mu: mu parameter of the distribution, equivalent to the mean.
    :param kappa: kappa parameter of the distribution. The larger kappa, the more concentrated are the points.
    Zero is equivalent to the uniform distribution.
    :param size: number of points.
    :return: an array of unit vectors in R**dim space.
    """

    result = []
    mu = np.array(mu, dtype=np.float64)
    if np.abs(np.linalg.norm(mu) - 1) > epsilon:
        raise ValueError(f'Mu must be a unit vector')

    c = np.exp(1)

    while len(result) < size:
        x = sample_uniform_on_sphere(rng, dim, 1)[0]
        n = np.linalg.norm(x)
        if n < epsilon:
            continue
        x /= n
        u = rng.uniform(0, c)
        f = np.exp(kappa * (np.dot(x, mu) - 1) + 1)
        if u <= f:
            result.append(x)

    return np.array(result)


class DimensionalityReduction:
    def __init__(self, x, accuracy=0.9):
        x = np.array(x)
        self._mean = x.mean(axis=0)
        x -= self._mean
        u, s, vt = np.linalg.svd(x)
        sum_s = np.sum(s)
        epsilon = 1e-5
        if sum_s > epsilon:
            dim_r = len(s) - (np.cumsum(s) / sum_s >= accuracy).sum() + 1
        else:
            dim_r = 1

        cov = s**2 / len(x)
        self.cov = cov[:dim_r]

        self._vr = vt.T[:, :dim_r]
        self._vt_back = vt[:dim_r, :]

    def reduce_dim(self, x):
        x = np.array(x) - self._mean
        xr = np.dot(x, self._vr)
        return xr

    def restore_dim(self, xr):
        xr = np.array(xr)
        x = np.dot(xr, self._vt_back)
        x += self._mean
        return x

