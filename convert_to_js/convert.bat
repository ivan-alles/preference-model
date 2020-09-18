rem Convert a TensorFlow model to a TensorFlow.js model.
rem Convert to dist folder, as the parcel server will search for files there.
rem Paramters:
rem %path to model

if not exist dist mkdir dist

tensorflowjs_converter %1 %1js