import { Engine } from '@/client-engine'

test('testfunc() returns a string "test"', () => {
  const engine = new Engine();
  expect(engine.testfunc()).toBe("test");
});