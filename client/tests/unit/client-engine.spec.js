import * as tf from '@tensorflow/tfjs';
tf.setBackend('cpu')

import { scaledDirichlet } from '@/client-engine'

test('tensorflow installed correctly', () => {
  const t = tf.tensor1d([1, 2, 3]);
  expect(t.shape).toStrictEqual([3]);
});

describe.each([
  [1000, 2, 1, 1],
  [1000, 2, 3, 1],
  [1000, 2, .5, 1],
  [1000, 2, .5, 0.1],
  [1000, 2, 1.5, 10],
  [1000, 3, 4, 1.5],
  [1000, 3, .4, 0.7],
  [1000, 10, .4, 0.7],
  [1000, 10, 4, 0.8],
])('scaledDirichlet(%i, %f, %f, %f)', (n, k, a, scale) => {
  const x = scaledDirichlet([n], k, a, scale);

  test('output is in expected range', () => {
    expect(x.shape).toStrictEqual([n, k]);
    expectTensorsClose(x.sum(1), tf.ones([n]));
    expectTensorsEqual(tf.greaterEqual(x, (1 - scale) / k), tf.ones([n, k]));
    expectTensorsEqual(tf.lessEqual(x, (scale * (k-1) + 1) / k), tf.ones([n, k]));
  });
  test('output has correct mean and std ', () => {
    expectTensorsClose(x.mean(0), tf.fill([k], 1 / k), 0.1);
    const stdExp = scale * Math.sqrt((k-1) / k**2 / (k * a + 1));
    expectTensorsClose(std(x, 0), tf.fill([k], stdExp), 0.1)
  });
});

/**
 * Check if tensors are close. Is needed as tf.test_util.expectArraysClose()
 * always succeedes with tensor arguments.
 *
 */
function expectTensorsClose(actual, expected, epsilon=null) {
  tf.test_util.expectArraysClose(
    actual.arraySync(),
    expected.arraySync(),
    epsilon);
}

/**
 * Check if tensors are close. Is needed as tf.test_util.expectArraysClose()
 * always succeedes with tensor arguments.
 *
 */
function expectTensorsEqual(actual, expected) {
  tf.test_util.expectArraysEqual(
    actual.arraySync(),
    expected.arraySync());
}

function std(x, axis=null) {
  const n = x.shape[axis];
  const m = tf.mean(x, axis)
  const xc = tf.sub(x, m)
  const x2 = tf.square(xc)
  return tf.sqrt(tf.div(tf.sum(x2, axis), n));
}