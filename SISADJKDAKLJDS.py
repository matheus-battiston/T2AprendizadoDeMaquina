import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K

# https://medium.com/@ashraf.dasa/bean-disease-classification-using-tensorflow-convolutional-neural-network-cnn-2079dffe87ce
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
checkpoint_dir = "./training_checkpoints"

PROJECT_NAME = "beans"
EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 6
EARLY_STOP_MONITOR = "loss"
NUM_FILTERS = 15
FILTER_SIZE = 10
POOL_SIZE = 4
DROP_PROBABILITY = 0.25
N_CLASSES = 3
LEARNING_RATE = 0.001
TAM = 224

RESNET = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"
MOBILENET_V3 = (
    "https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/classification/5"
)
INCEPTION = "https://tfhub.dev/google/tf2-preview/inception_v3/classification/4"
INCEPTION_RESNET = (
    "https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/5"
)
INATURALIST = "https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/5"


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))


def recall(y_true, y_pred):
    # recall of class 1

    # do not use "round" here if you're going to use this as a loss function
    true_positives = K.sum(K.round(y_pred) * y_true)
    possible_positives = K.sum(y_true)
    return true_positives / (possible_positives + K.epsilon())


lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    LEARNING_RATE, decay_steps=33 * 1000, decay_rate=1, staircase=False
)


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(y_pred) * y_true)
    predicted_positives = K.sum(y_pred)
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras


def resize_and_rescale(image, label):
    image = tf.image.resize(image, [TAM, TAM])
    return image, label


def retrive_data():
    test_ds, info = tfds.load(
        PROJECT_NAME,
        split="test",
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
    )
    print(info)
    # to see labels
    print(f'Classes:{info.features["label"].names}')
    # show the shape
    print(test_ds.element_spec)
    return test_ds, info


def get_training_data():
    validation_data = tfds.load(PROJECT_NAME, split=f"validation", as_supervised=True)
    training_data = tfds.load(PROJECT_NAME, split=f"train", as_supervised=True)
    return training_data, validation_data


def wrangle_data_GenPlus(dataset, split, batch_size):
    wrangled = dataset.map(lambda img, lbl: (tf.cast(img, tf.float32) / 255.0, lbl))
    if split:
        features = np.array([x[0] for x in wrangled])
        lables = np.array([x[1] for x in wrangled])
        train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip=True, zoom_range=0.2, rotation_range=20, fill_mode="nearest"
        )
        wrangled = train_data_gen.flow(features, lables, batch_size=batch_size)
    else:  # Caches the elements in this dataset. loat it into the memory to go faster
        wrangled = wrangled.cache()
        wrangled = wrangled.batch(
            batch_size
        )  # Combines consecutive elements of this dataset into batches.
        wrangled = wrangled.prefetch(tf.data.AUTOTUNE)

    return wrangled


def compileModel(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy", recall, precision, f1],
    )
    return model


def buildModel(layers):
    neural_net = tf.keras.Sequential(
        [
            layers,
            tf.keras.layers.Dropout(DROP_PROBABILITY),
            tf.keras.layers.Dense(N_CLASSES, activation="softmax"),
        ]
    )
    return compileModel(neural_net)


def plot_History(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


def execModel(link):
    layers = hub.KerasLayer(link, input_shape=(TAM, TAM, 3))
    layers.trainable = False

    epochs_range = range(EPOCHS)
    model = buildModel(layers)
    history = model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS)

    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()

    plot_History(history)
    print(model.evaluate(test_ds))


if __name__ == "__main__":
    test_ds, info = retrive_data()
    train_ds, valid_ds = get_training_data()
    train_ds = train_ds.map(
        resize_and_rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    valid_ds = valid_ds.map(
        resize_and_rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_ds = test_ds.map(
        resize_and_rescale, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    train_ds = wrangle_data_GenPlus(train_ds, True, batch_size=BATCH_SIZE)
    valid_ds = wrangle_data_GenPlus(valid_ds, False, batch_size=BATCH_SIZE)
    test_ds = wrangle_data_GenPlus(test_ds, False, batch_size=BATCH_SIZE)

    callback = tf.keras.callbacks.EarlyStopping(
        monitor=EARLY_STOP_MONITOR, patience=PATIENCE
    )

    # execModel(INCEPTION)
    # execModel(RESNET)
    # execModel(INCEPTION_RESNET)
    execModel(INATURALIST)
    # execModel(MOBILENET_V3)
