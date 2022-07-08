import csv
from matplotlib import image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


####################################################################
### Training policy
####################################################################
bs = 128
lr = 0.001
num_epochs=20

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

####################################################################
### Data
####################################################################
SAVE_PATH = "./result"
TRAIN_FILE = "./mnist/sign_mnist_train.csv"
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

train_images, train_labels = parse_data_from_file(TRAIN_FILE)
train_images = np.expand_dims(train_images, axis=3)

test_images, test_labels = parse_data_from_file(TEST_FILE)
test_images = np.expand_dims(test_images, axis=3)



####################################################################
### ImageDataGenerator
####################################################################
transform_train = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    fill_mode="nearest"
)
data_train = transform_train.flow(x=train_images, y=train_labels, batch_size=bs)

transform_test = ImageDataGenerator(rescale=1.0/255.0)
data_test = transform_test.flow(x=test_images, y=test_labels, batch_size=bs)


####################################################################
### Model
####################################################################

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(26, activation="sigmoid"),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    loss = "sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    data_train,
    epochs=num_epochs,
    validation_data=data_test
)



####################################################################
### Save model
####################################################################
model.save(SAVE_PATH)