import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


####################################################################
### Data
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
### TF native model
####################################################################
TF_RESULT= "./tf_result"
tf_model = tf.keras.models.load_model(TF_RESULT)

print("TF native model start...")
for idx, (x, result) in enumerate(data_test):
    y = tf_model.predict(x)
    if idx > len(test_images):
        break
    

####################################################################
### TRT accelerated model
####################################################################

TRT_RESULT= "./trt_result"

trt_model = tf.saved_model.load(
    TRT_RESULT, tags=['serve'])
infer = trt_model.signatures['serving_default']

print("TRT accelerated model start...")
for idx, (x, result) in enumerate(data_test):
    x = tf.constant(x)
    y = infer(x)
    print(y, result)
    if idx > len(test_images):
        break
    
