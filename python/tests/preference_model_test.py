# Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import numpy as np
from server import preference_model
import matplotlib.pyplot as plt


def test_spherical_to_cartesian_n_2():
    def compute(r, phi):
        exp = [
            r * np.cos(phi[0]),
            r * np.sin(phi[0])
        ]
        return exp

    r = 1.
    phi = np.array([0.1])
    act = preference_model.spherical_to_cartesian_n(r, phi)
    exp = compute(r, phi)
    assert np.allclose(exp, act)

    r = 0.
    phi = np.array([0.1])
    act = preference_model.spherical_to_cartesian_n(r, phi)
    exp = compute(r, phi)
    assert np.allclose(exp, act)

    r = 0.
    phi = np.array([0.2])
    act = preference_model.spherical_to_cartesian_n(r, phi)
    exp = compute(r, phi)
    assert np.allclose(exp, act)

    r = 3.
    phi = np.array([0.1])
    act = preference_model.spherical_to_cartesian_n(r, phi)
    exp = compute(r, phi)
    assert np.allclose(exp, act)

    r = 2.
    phi = np.array([0.3])
    act = preference_model.spherical_to_cartesian_n(r, phi)
    exp = compute(r, phi)
    assert np.allclose(exp, act)

    r = 2.
    phi = np.array([-0.3])
    act = preference_model.spherical_to_cartesian_n(r, phi)
    exp = compute(r, phi)
    assert np.allclose(exp, act)


def test_spherical_to_cartesian_n_3():
    def compute(r, phi):
        exp = [
            r * np.cos(phi[0]),
            r * np.sin(phi[0]) * np.cos(phi[1]),
            r * np.sin(phi[0]) * np.sin(phi[1])
        ]
        return np.array(exp).ravel()
    r = 1.
    phi = np.array([0.1, 0.2])
    act = preference_model.spherical_to_cartesian_n(r, phi)
    exp = compute(r, phi)
    assert np.allclose(exp, act)

    r = 2.
    phi = np.array([0.1, 0.2])
    act = preference_model.spherical_to_cartesian_n(r, phi)
    exp = compute(r, phi)
    assert np.allclose(exp, act)

    r = np.array([3., 4., 5]).reshape(-1, 1)
    phi = np.array([[0.1, -0.2], [0.1, 0.2], [0.3, -0.2]])
    act = preference_model.spherical_to_cartesian_n(r, phi)
    exp0 = compute(r[0], phi[0])
    exp1 = compute(r[1], phi[1])
    exp2 = compute(r[2], phi[2])
    assert np.allclose(exp0, act[0])
    assert np.allclose(exp1, act[1])
    assert np.allclose(exp2, act[2])


def convert_cartesian_to_spherical(x):
    r = np.linalg.norm(x)
    n = len(x)
    phi = np.zeros(n - 1)
    k = -1
    for i in range(n - 1, -1, -1):
        if x[i] != 0:
            k = i
            break
    if k == -1:
        return r, phi

    for i in range(0, k):
        phi[i] = np.arccos(x[i] / np.linalg.norm(x[i:]))

    if k == n - 1:
        phi[n - 2] = np.arccos(x[n-2] / np.linalg.norm(x[-2:]))
        if x[n-1] < 0:
            phi[n - 2] *= -1
    else:
        phi[k] = 0 if x[k] > 0 else np.pi

    return r, phi


def test_cartesian_to_spherical_n_2():
    x = np.array([
        [0, 0],
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -2],
        [2, 3],
        [-1, 4]
    ], dtype=np.float32)

    r_act, phi_act = preference_model.cartesian_to_spherical_n(x)

    for i in range(len(x)):
        r_exp, phi_exp = convert_cartesian_to_spherical(x[i])
        assert np.allclose(r_act[i], r_exp)
        assert np.allclose(phi_act[i], phi_exp)


def test_cartesian_to_spherical_n_3():
    x = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [2, 1, 0],
        [2, -1, 0],
        [-2, -1, 0],
        [2, 3, 1],
        [2, 3, -1],
        [2, 0, 5],
        [0, 0, 1],
        [0, 0, -2]
    ], dtype=np.float32)

    r_act, phi_act = preference_model.cartesian_to_spherical_n(x)

    for i in range(len(x)):
        r_exp, phi_exp = convert_cartesian_to_spherical(x[i])
        assert np.allclose(r_act[i], r_exp)
        assert np.allclose(phi_act[i], phi_exp)


def test_cartesian_to_spherical_n_4():
    x = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [-1, 0, 0, 0],
        [1, 2, 0, 0],
        [-1, -2, 0, 0],
        [2, 2, 1, 0],
        [-1, -2, -3, 0],
        [2, 2, 1, -1],
        [-1, -2, -3, 4],
    ], dtype=np.float32)

    r_act, phi_act = preference_model.cartesian_to_spherical_n(x)

    for i in range(len(x)):
        r_exp, phi_exp = convert_cartesian_to_spherical(x[i])
        assert np.allclose(r_act[i], r_exp)
        assert np.allclose(phi_act[i], phi_exp)


def test_cartesian_to_spherical_n_back_and_forth():
    rng = np.random.RandomState(seed=1)

    for r in range(10000):
        n = rng.randint(2, 100)
        if rng.uniform(0, 1) < .1:
            x = rng.uniform(-10, 10, n)  # row-vector
        else:
            m = rng.randint(1, 10)
            x = rng.uniform(-10, 10, (m, n))  # array of row vectors
        r, phi = preference_model.cartesian_to_spherical_n(x)
        if phi.ndim == 1:
            phi = phi.reshape(1, -1)

        assert phi.shape[1] == n - 1
        # Last phi is in range -pi, pi
        assert np.logical_and(phi[:, -1] >= -np.pi, phi[:, -1] <= np.pi).all()
        # Other phi are in range [0, pi]
        if n > 2:
            assert np.logical_and(phi[:, :-1] >= 0, phi[:, :-1] <= np.pi).all()

        x1 = preference_model.spherical_to_cartesian_n(r, phi)
        assert np.allclose(x, x1)

        # Make sure that spherical -> cartesian transform is invariant to
        # the following angle transformations:

        phi[:, -1] += rng.randint(-10, 10) * 2 * np.pi

        x2 = preference_model.spherical_to_cartesian_n(r, phi)
        assert np.allclose(x, x2)


def test_mean_of_angles():
    m_act = preference_model.mean_of_angles(np.array([0.0, 0, 0]))
    assert np.allclose(0, m_act)

    m_act = preference_model.mean_of_angles(np.array([0.1, 0.1, 0.1, 0.1]))
    assert np.allclose(0.1, m_act)

    m_act = preference_model.mean_of_angles(np.array([0.1, -0.1, 0.1, -0.1]))
    assert np.allclose(0.0, m_act)

    m_act = preference_model.mean_of_angles(np.array([0, 2*np.pi]))
    assert np.allclose(0.0, m_act)

    m_act = preference_model.mean_of_angles(np.array([0, 2 * np.pi, -10 * np.pi]))
    assert np.allclose(0.0, m_act)

    m_act = preference_model.mean_of_angles(np.array([
        [0.1, 0.1 + 2 * np.pi, 0.1 - 10 * np.pi],
        [-0.1, -0.1 + 2 * np.pi, -0.1 - 10 * np.pi]
    ]), axis=1)
    assert np.allclose([0.1, -0.1], m_act)

    m_act = preference_model.mean_of_angles(np.array([
        [0.1, -0.1 + 2 * np.pi, -0.1 - 10 * np.pi, 0.1 + 8 * np.pi],
        [np.pi, 3*np.pi, -5*np.pi, 9*np.pi],
    ]), axis=1)
    assert np.allclose([0, np.pi], m_act)


def test_sample_uniform_on_sphere():
    # Generate a n-sphere and make sure that there are a roughly
    # equal number of points around each of a chosen vectors.
    rng = np.random.RandomState(seed=1)
    dim = 10
    num_points = 2000000
    points = preference_model.sample_uniform_on_sphere(rng, dim, num_points)
    points /= np.linalg.norm(points, axis=-1, keepdims=True)

    vectors = np.eye(dim)

    dist = np.dot(points, vectors)
    cos_threshold = 0.8  # Search for points such that cos(v, p) > cos_threshold

    close_points = (dist > cos_threshold).sum(axis=0)
    close_points_ratio = close_points / num_points
    close_points_ratio /= close_points_ratio.mean()
    for i in range(len(close_points_ratio)):
        assert np.abs(close_points_ratio[i] - 1) < 0.1


def test_sample_uniform_on_sphere_plot():
    # Set to False to see the plot
    if True:
        return

    rng = np.random.RandomState(seed=1)
    points = preference_model.sample_uniform_on_sphere(rng, 3, 1000)
    points /= np.linalg.norm(points, axis=-1, keepdims=True)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    points = np.rollaxis(points, 1, 0)
    ax.scatter(*points)
    plt.show()


def test_scaled_dirichlet_plot():
    # Set to False to see the plot
    if True:
        return

    rng = np.random.RandomState(seed=1)
    points = preference_model.scaled_dirichlet(rng, k=3, a=1, size=1000, scale=1)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    points = np.rollaxis(points, 1, 0)
    ax.scatter(*points, marker='+')
    r = [-0.2, 1.2]
    ax.set_xlim(*r)
    ax.set_ylim(*r)
    ax.set_zlim(*r)
    plt.show()

def test_scaled_dirichlet():
    rng = np.random.RandomState(1)

    for i in range(100):
        k = rng.randint(2, 10)
        a = rng.uniform(.1, 10)
        scale = rng.uniform(.5, 2)
        n = 5000
        x = preference_model.scaled_dirichlet(rng, k, a, n, scale)
        assert x.shape == (n, k)
        assert np.allclose(x.sum(axis=1), np.ones(n))
        assert (x >= (1-scale) / k).all()
        assert (x <= (scale * (k-1) + 1) / k).all()
        assert np.allclose(x.mean(axis=0), np.full(k, 1/k), rtol=0.1)
        std = scale * np.sqrt((k-1) / k**2 / (k * a + 1))
        assert np.allclose(x.std(axis=0), np.full(k, std), rtol=0.1)

