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
    

  
  
## Reference
[1] https://www.kaggle.com/datasets/datamunge/sign-language-mnist [accessed July 17, 2022]