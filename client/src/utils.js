/* eslint-disable no-unused-vars */

function float32ArrayToBase64(farr) {
  // See https://gist.github.com/sketchpunk/6c60f6b78d4b66c729dcbf460ea06b42
  const tarr = new Float32Array(farr);
	const uint = new Uint8Array( tarr.buffer );
  const str = btoa( String.fromCharCode.apply( null, uint ) );
  return str;
}

function base64ToFloat32Array(str) {
  let bytes = atob(str);
	let buf = new ArrayBuffer( bytes.length );
	let dataView = new DataView( buf );
  for( let i=0; i < bytes.length; i++ ) {
    dataView.setUint8( i, bytes.charCodeAt(i) );
  }
  let tarr = new Float32Array(buf);
  return Array.from(tarr);
}

export { float32ArrayToBase64, base64ToFloat32Array }