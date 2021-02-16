import * as React from 'react';
import * as tfjs from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import cat from './imgs/cat.jpg';
// import dog from './imgs/dog.jpeg';
import './App.css';

function App() {

  const predictImage = async () => {
    const img = document.getElementById('image');
    console.log("模型加载中...");
    const model = await mobilenet.load();
    console.log("模型加载完毕!")
    const predictions = await model.classify(img);
    console.log('预测结果: ', predictions);
  }

  React.useEffect(() => {
    predictImage();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <img id="image" alt="cat lay down" src={cat} />
      </header>
    </div>
  );
}

export {
  App,
  tfjs
};
