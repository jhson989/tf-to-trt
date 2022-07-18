import os, csv, time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



####################################################################
### Set up GPU environment
####################################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)



####################################################################
### Load data
####################################################################
TEST_FILE = "./mnist/sign_mnist_test.csv"

def parse_data_from_file(FILE_NAME):

    labels = []
    images = []
    with open(FILE_NAME) as file:
        csv_reader = csv.reader(file, delimiter=",")
        next(csv_reader, None)
        for line in csv_reader:
            labels.append(line[0])
            img = np.reshape(line[1:], (28, 28)).tolist()
            images.append(img)
        
    return np.ndarray.astype(np.array(images), "float64"), np.ndarray.astype(np.array(labels), "float64")

test_images, test_labels = parse_data_from_file(TEST_FILE)
test_images = np.expand_dims(test_images, axis=3)

transform_test = ImageDataGenerator(rescale=1.0/255.0)
data_test = transform_test.flow(x=test_images, y=test_labels, batch_size=1)



####################################################################
### Load models
####################################################################
TF_MODEL_PATH= "./tf_result"
tf_model = tf.keras.models.load_model(TF_MODEL_PATH)

TRT_MODEL_PATH= "./trt_result"
trt_model = tf.saved_model.load(TRT_MODEL_PATH, tags=['serve'])
trt_infer = trt_model.signatures['serving_default']



####################################################################
### Inference time test : TF native model
####################################################################
#print("\n\n==========================================================")
#print("TF native model started...")
#start_time = time.time()
#for idx, (x, result) in enumerate(data_test):
#    y = tf_model.predict(x)
#    if idx > len(test_images):
#        break
#end_time = time.time()
#print(" -- elapsed time : %.3f s [%d images]" % (end_time-start_time, len(test_images)))
#


####################################################################
### Inference time test : TRT accelerated model
####################################################################
#print("\n\n==========================================================")
#print("TRT accelerated model started...")
#start_time = time.time()
#for idx, (x, result) in enumerate(data_test):
#    x = tf.constant(x)
#    y = trt_infer(x)
#    if idx > len(test_images):
#        break
#end_time = time.time()
#print(" -- elapsed time : %.3f s [%d images]" % (end_time-start_time, len(test_images)))
#


####################################################################
### Inference accuracy test
####################################################################
print("\n\n==========================================================")
print("Test for inference accuracy check started...")
for idx, (x, result) in enumerate(data_test):
    tf_y = tf_model.predict(x)
    trt_x = tf.constant(x)
    trt_y = trt_infer(trt_x)["dense_1"].numpy()

    if (tf_y != trt_y).all():
        print(tf_y)
        print(trt_y)
        print("Error occurred!!")
        break

    if idx > len(test_images):
        print("No inference error")
        break