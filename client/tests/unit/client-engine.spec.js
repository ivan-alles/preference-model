import * as tf from '@tensorflow/tfjs';
tf.setBackend('cpu')

import { scaledDirichlet } from '@/client-engine'

test('tensorflow installed correctly', () => {
  const t = tf.tensor1d([1, 2, 3]);
  expect(t.shape).toStrictEqual([3]);
});

test('scaledDirichlet() shall work with various random parameters', () => {

  const k = 2;
  const a = 3;
  const scale = 1;
  const n = 5000;
  const x = scaledDirichlet(k, a, [n], scale);

  expect(x.shape).toStrictEqual([n, k]);

});