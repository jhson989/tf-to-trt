# TensorFlow to TensorRT

A sample program for how to accelerate deep learning inference via TensorRT
The sample model is trained on sign language MNIST[1] dataset.
  

## 0. Prerequisite
- Test Enviroment
    - OS
        - Ubuntu 18.04
    - NVIDIA
        - Driver 470.129
        - CUDA 10.1
        - cuDNN 7.6
    - python 3.7
        - TensorFlow 2.2

## 1. How to Run
- Prepare data for training & inferencing
    - mkdir mnist && cd mnist
    - [download "sign_mnist_train.csv" and "sign_mnist_test.csv" from [1]]
- Train and save a TensorFlow model
    - python trainer.py [result: tf_result/model.pb]
- Convert a saved TF model to a TRT model
    - python converter.py [result: trt_result/model.pb]
- Compare inference performance bewteen the TR and TRT models
    - python tester.py


## 2. Sample result
- Inference performance
    - Total 7172 test images 
    - TF naive model (GPU): 105.116 s
    - TRT accelerated model: 2.5212 s
- Inference accuracy
    - TRT accelerated model predicts well with relative error < 0.001 (0.1%) and absolute error < 0.0001
  
  
## Reference
[1] https://www.kaggle.com/datasets/datamunge/sign-language-mnist [accessed July 17, 2022]