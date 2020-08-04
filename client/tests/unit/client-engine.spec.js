import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node-gpu';
import { scaledDirichlet } from '@/client-engine'


test('scaledDirichlet() works', () => {

  const k = 2;
  const a = 3;
  const scale = 1;
  const n = 5000;
  const x = scaledDirichlet(k, a, [n], scale);

  const dummy = tf.tensor1d([1, 2, 3]);

  expect(x.shape).toBe([n, k]);
  expect(dummy).toBe([n, k]);

});