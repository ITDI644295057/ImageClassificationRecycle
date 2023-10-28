const model_path = "models/";

let model, labels, labelContainer, maxPredictions;

// Load the image model and setup the webcam
async function init() {
  labels = ["Plastic_Bottles","Paper","Aluminum_Cans"]; // Replace with actual class labels
  maxPredictions=labels.length;

  model = await tflite.loadTFLiteModel("./models/model_unquant.tflite");

  labelContainer = document.getElementById("label-container");
  for (let i = 0; i < maxPredictions; i++) {
    // and class labels
    labelContainer.appendChild(document.createElement("div"));
  }
  
}

async function imageShow() {
  const inputElement = document.getElementById("imageFileInput");
  const image = inputElement.files[0];
  const imgElement = document.getElementById("imageInput");
  imgElement.src = URL.createObjectURL(image);
  imgElement.srcObject = image;

  for (let i = 0; i < maxPredictions; i++) {
    labelContainer.childNodes[i].innerHTML = "";

  }
  btnpredict.hidden=false
}

async function predictImage() {
  // Load and preprocess the uploaded image
  const imgElement = document.querySelector("img");

  const tensor = tf.browser.fromPixels(imgElement).toFloat();
  const resizedTensor = tf.image
    .resizeBilinear(tensor, [224, 224])
    .expandDims()
    .div(127.5)
    .sub(1);
  const preprocessedTensor = resizedTensor.expandDims(0);

  // Run inference on the preprocessed image
  const predictions = await model.predict(resizedTensor);

  for (let i = 0; i < maxPredictions; i++) {
      const classPrediction =labels[i] + " : " + predictions.dataSync()[i].toFixed(2);
      labelContainer.childNodes[i].innerHTML = classPrediction;

  }

  btnpredict.hidden=true

  // Dispose of tensors to free up memory
  tensor.dispose();
  resizedTensor.dispose();
  preprocessedTensor.dispose();
}

init();
