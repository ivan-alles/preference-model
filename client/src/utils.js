/* eslint-disable no-unused-vars */

const FACTOR = 99;
const ELEMENT_SIZE = 2;

function float32ArrayToBase64(farr) {
  // See https://gist.github.com/sketchpunk/6c60f6b78d4b66c729dcbf460ea06b42
  let result = '';
  for(let i = 0; i < farr.length; ++i) {
    let element = Math.round((farr[i] / Math.PI * 0.5 + 0.5) * FACTOR);
    element = Math.min(Math.max(0, element), FACTOR);
    console.log(i, farr[i], element);
    result += String(element).padStart(ELEMENT_SIZE, '0'); 
  }
  return result;
}

function base64ToFloat32Array(str) {
  let farr = new Array();
  for(let i = 0; i < str.length; i += ELEMENT_SIZE) {
    const element = (str.substr(i, 2) / FACTOR - 0.5) * 2 * Math.PI;
    console.log(i / ELEMENT_SIZE, element);
    farr.push(element);
  }
  return farr;
}

export { float32ArrayToBase64, base64ToFloat32Array }