import { float32ArrayToBase64, base64ToFloat32Array } from '@/utils'


describe('Convert float array to base64 and back', () => {
  const farr = [1.23, -132.2, 23.0];
  
  const str = float32ArrayToBase64(farr);

  test('A string is returned', () => {
    expect(typeof(str)).toStrictEqual('string');
  });

  const farr1 = base64ToFloat32Array(str);

  test('The new and original arrays have the same length', () => {
    expect(farr1.length).toBeCloseTo(farr.length);
  });

  for (let i = 0; i < farr.length; ++i) {
    test(`The new and original array at pos ${i} are close `, () => {
      expect(farr1[i]).toBeCloseTo(farr[i]);
    });
  }

});

