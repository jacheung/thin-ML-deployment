import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from app.ml import preprocessing


class Model:
    def __init__(self, model_path: str = None):
        self._model = None
        self._model_path = model_path
        self.load()

    def train(self, xy_tuple):
        # # GPU acceleration if possible. First check to make sure you have GPUs available.
        # # will need to pip install tensorflow-metal
        # gpus = tf.config.list_logical_devices('GPU')
        # strategy = tf.distribute.MirroredStrategy(gpus)
        # with strategy.scope():
        learning_rate = 0.01
        l1 = 0.
        l2 = 0.
        num_hidden = 16
        regularizer = tf.keras.regularizers.l1_l2(l1, l2)

        layers = [
            hub.KerasLayer(
                "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
                input_shape=(224, 224, 3),
                trainable=False,
                name='mobilenet_embedding'),
            tf.keras.layers.Dense(num_hidden,
                                  kernel_regularizer=regularizer,
                                  activation='relu',
                                  name='dense_hidden'),
            tf.keras.layers.Dense(len(class_names),
                                  kernel_regularizer=regularizer,
                                  activation='softmax',
                                  name='mnist_prob')
        ]
        self._model = tf.keras.Sequential(layers, name='mnist_classification')
        self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=False),
                            metrics=['accuracy'])
        self._model.fit(xy_tuple, epochs=5)
        return self

    def predict_single_image(self, image: np.ndarray) -> np.ndarray:
        image, _ = preprocessing.preprocess_mnist_tfds(image)
        image = tf.reshape(image, [1, 224, 224, 3])
        return self._model.predict(image).argmax()

    def save(self):
        if self._model is not None:
            ts = int(time.time())
            self._model.save(filepath=f'{self._model_path}/{ts}', save_format='tf')
        else:
            raise TypeError("The model is not trained yet, use .train() before saving")

    def load(self):
        try:
            latest = str(max([int(x) for x in os.listdir(self._model_path)]))
            self._model = tf.keras.models.load_model(f'{self._model_path}/{latest}')
        except FileNotFoundError:
            print('No models found.')
            self._model = None
        return self


top_level_dir = os.path.abspath(os.path.dirname(__file__))
file_path = f"{top_level_dir}/img_classifier/"
model = Model(file_path)


def get_model():
    return model


if __name__ == "__main__":
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ds_train = ds_train.map(preprocessing.preprocess_mnist_tfds, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(128)
    model.train(ds_train)
    model.save()
