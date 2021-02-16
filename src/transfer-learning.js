import * as React from 'react';
import * as tfjs from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';
// import cat from './imgs/cat.jpg';
// import dog from './imgs/dog.jpeg';
import './App.css';



// Number of classes to classify
const NUM_CLASSES = 2;
// Webcam Image size. Must be 227.
const IMAGE_SIZE = 227;
// K value for KNN
const TOPK = 10;

// const classes = ["Left", "Right"];
let testPrediction = false;
let training = true;
let video;

class App extends React.Component {
  constructor(props) {
    super(props);
    this.infoTexts = [];
    this.training = -1; // -1 when no class is being trained
    this.recordSamples = false;
    this.knn = undefined;
    this.mobilenetModule = undefined;
    this.timer = undefined;

    this.loadClassifierAndModel = this.loadClassifierAndModel.bind(this);
    this.initiateWebcam = this.initiateWebcam.bind(this);
    this.setupButtonEvents = this.setupButtonEvents.bind(this);
  }

  async loadClassifierAndModel() {
    console.log('模型加载中======>');
    this.knn = knnClassifier.create();
    this.mobilenetModule = await mobilenet.load();
    console.log("<======模型加载完毕");

    this.start();
  }

  initiateWebcam() {
    // Setup webcam
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then(stream => {
        video.srcObject = stream;
        video.width = IMAGE_SIZE;
        video.height = IMAGE_SIZE;
      });
  }

  setupButtonEvents() {
    for (let i = 0; i < NUM_CLASSES; i++) {
      let button = document.querySelectorAll(".button")[i];

      button.onmousedown = () => {
        console.log('点击了采集按钮')
        this.training = i;
        this.recordSamples = true;
      };
      button.onmouseup = () => (this.training = -1);

      const infoText = document.querySelectorAll(".info-text")[i];
      infoText.innerText = " No examples added";
      this.infoTexts.push(infoText);
    }
  }

  start() {
    if (this.timer) {
      this.stop();
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }

  stop() {
    cancelAnimationFrame(this.timer);
  }

  async animate() {
    if (this.recordSamples) {
      // Get image data from video element
      const image = tfjs.browser.fromPixels(video);

      let logits;
      // 'conv_preds' is the logits activation of MobileNet.
      const infer = () => this.mobilenetModule.infer(image, "conv_preds");

      // Train class if one of the buttons is held down
      if (this.training !== -1) {
        logits = infer();

        // Add current image to classifier
        this.knn.addExample(logits, this.training);
      }

      const numClasses = this.knn.getNumClasses();

      if (testPrediction) {
        training = false;
        if (numClasses > 0) {
          // If classes have been added run predict
          logits = infer();
          const res = await this.knn.predictClass(logits, TOPK);

          for (let i = 0; i < NUM_CLASSES; i++) {
            // The number of examples for each class
            const exampleCount = this.knn.getClassExampleCount();

            // Make the predicted class bold
            if (res.classIndex === i) {
              this.infoTexts[i].style.fontWeight = "bold";
            } else {
              this.infoTexts[i].style.fontWeight = "normal";
            }

            if (exampleCount[i] > 0) {
              this.infoTexts[i].innerText = ` ${
                exampleCount[i]
              } examples - ${res.confidences[i] * 100}%`;
            }
          }
        }
      }

      if (training) {
        // The number of examples for each class
        const exampleCount = this.knn.getClassExampleCount();

        for (let i = 0; i < NUM_CLASSES; i++) {
          if (exampleCount[i] > 0) {
            this.infoTexts[i].innerText = ` ${exampleCount[i]} examples`;
          }
        }
      }

      // Dispose image when done
      image.dispose();

      if (logits != null) {
        logits.dispose();
      }
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }

  componentDidMount = () => {
    video = document.querySelector("#webcam");
    // Initiate deeplearn.js math and knn classifier objects
    this.loadClassifierAndModel();
    this.initiateWebcam();
    this.setupButtonEvents();

    document
    .querySelector(".test-predictions")
    .addEventListener("click", function() {
        console.log('点击了预测按钮')
      testPrediction = true;
    });
  }

  render() {
      return (
        <main>
            <section className="button-section left">
                <button className="button">Left</button>
                <div className="examples-text">
                    <span className="info-text"></span>
                </div>
            </section>

            <video autoPlay id="webcam" width="224" height="224"></video>

            <section className="button-section right">
                <button className="button">Right</button>
                <div className="examples-text">
                    <span className="info-text"></span>
                </div>
            </section>

            <section className="button-section test">
                <button className="button test-predictions">Test predictions</button>
            </section>
        </main>
      )
  }
}

export {
  App,
  tfjs
};
