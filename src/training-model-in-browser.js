import * as React from 'react';
import * as tf from "@tensorflow/tfjs";
import trainingSet from "./training.json";
import testSet from "./testing.json";
import './App.css';

function App() {

  let trainingData, testingData, outputData, model;
  let training = true;
  let predictButton;

  const init = async () => {
    splitData();
    createModel();
    await trainData();
    if (!training) {
      predictButton.disabled = false;
      predictButton.onclick = () => {
        const inputData = getInputData();
        predict(inputData);
      };
    }
  };

  const splitData = () => {
    trainingData = tf.tensor2d(
      trainingSet.map(item => [
        item.sepal_length,
        item.sepal_width,
        item.petal_length,
        item.petal_width
      ]),
      [130, 4]
    );

    testingData = tf.tensor2d(
      testSet.map(item => [
        item.sepal_length,
        item.sepal_width,
        item.petal_length,
        item.petal_width
      ]),
      [14, 4]
    );

    outputData = tf.tensor2d(
      trainingSet.map(item => [
        item.species === "setosa" ? 1 : 0,
        item.species === "virginica" ? 1 : 0,
        item.species === "versicolor" ? 1 : 0
      ]),
      [130, 3]
    );
  };

  const createModel = () => {
    model = tf.sequential();
    model.add(
      tf.layers.dense({ inputShape: 4, activation: "sigmoid", units: 10 })
    );

    model.add(
      tf.layers.dense({
        inputShape: 10,
        units: 3,
        activation: "softmax"
      })
    );

    model.compile({
      loss: "categoricalCrossentropy",
      optimizer: tf.train.adam()
    });
  };

  const trainData = async () => {
    let numSteps = 15;
    let trainingStepsDiv = document.getElementsByClassName("training-steps")[0];
    for (let i = 0; i < numSteps; i++) {
      let res = await model.fit(trainingData, outputData, { epochs: 40 });
      trainingStepsDiv.innerHTML = `训练过程: ${i}/${numSteps - 1}, 数据丢失: ${
        res.history.loss[0]
      }`;
      if (i === numSteps - 1) {
        training = false;
      }
    }
  };

  const predict = async inputData => {
    for (let [key, value] of Object.entries(inputData)) {
      inputData[key] = parseFloat(value);
    }
    inputData = [inputData];

    let newDataTensor = tf.tensor2d(
      inputData.map(item => [
        item.sepal_length,
        item.sepal_width,
        item.petal_length,
        item.petal_width
      ]),
      [1, 4]
    );

    let prediction = model.predict(newDataTensor);

    displayPrediction(prediction);
  };

  const getInputData = () => {
    let sepalLength = document.getElementsByName("sepal-length")[0].value;
    let sepalWidth = document.getElementsByName("sepal-width")[0].value;
    let petalLength = document.getElementsByName("petal-length")[0].value;
    let petalWidth = document.getElementsByName("petal-width")[0].value;

    return {
      sepal_length: sepalLength,
      sepal_width: sepalWidth,
      petal_length: petalLength,
      petal_width: petalWidth
    };
  };

  const displayPrediction = prediction => {
    let predictionDiv = document.getElementsByClassName("prediction")[0];
    let predictionSection = document.getElementsByClassName(
      "prediction-block"
    )[0];

    let maxProbability = Math.max(...prediction.dataSync());
    let predictionIndex = prediction.dataSync().indexOf(maxProbability);
    let irisPrediction;

    switch (predictionIndex) {
      case 0:
        irisPrediction = "Setosa";
        break;
      case 1:
        irisPrediction = "Virginica";
        break;
      case 2:
        irisPrediction = "Versicolor";
        break;
      default:
        irisPrediction = "";
        break;
    }
    predictionDiv.innerHTML = irisPrediction;
    predictionSection.style.display = "block";
  };

  React.useEffect(() => {
    predictButton = document.querySelector(".predict");
    init();
  }, []);


  return (
    <div>
      <section className="data-inputs">
        <h3>鸢尾花分类</h3>
        <p>模型训练中...</p>
        <p className="training-steps" />
        <div className="input-block">
          <label htmlFor="sepal-length">萼片长度(sepal_length):</label>
          <input name="sepal-length" type="number" min="0" max="100" placeholder="1.5" />
        </div>

        <div className="input-block">
          <label htmlFor="sepal-width">萼片宽度(sepal_width):</label>
          <input name="sepal-width" type="number" min="0" max="100" placeholder="0.4" />
        </div>

        <div className="input-block">
          <label htmlFor="petal-length">花瓣长度(petal_length):</label>
          <input name="petal-length" type="number" min="0" max="100" placeholder="1.0" />
        </div>

        <div className="input-block">
          <label htmlFor="petal-width">花瓣宽度(petal_width):</label>
          <input name="petal-width" type="number" min="0" max="100" placeholder="0.7" />
        </div>

        <button className="predict" disabled>点击预测</button>
      </section>

      <section className="prediction-block">
        <p>鸢尾花预测分类结果:</p>
        <p className="prediction"></p>
      </section>
    </div>
  );
}

export {
  App
}