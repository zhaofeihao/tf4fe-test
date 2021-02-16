## 启动
npm install
npm start

## 测试
分别测试三种方法时需要在index.js中引用不同的文件
``` javascript
import { App } from './pre-trained-model';
import { App } from './transfer-learning';
import { App } from './training-model-in-browser';
```

## note
模型运行速度有点慢，需要耐心等待