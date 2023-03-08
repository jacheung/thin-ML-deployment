import joblib
import numpy as np
from pathlib import Path

import tensorflow_hub as hub
import tensorflow as tf
import preprocessing


class Model:
    def __init__(self, model_path: str = None):
        self._model = None
        self._model_path = model_path
        self.load()

    def train(self, X: np.ndarray, y: np.ndarray):
        learning_rate = 0.001
        l1 = 0.
        l2 = 0.
        num_hidden = 16
        regularizer = tf.keras.regularizers.l1_l2(l1, l2)
        class_names = np.unique(y)

        layers = [
            hub.KerasLayer(
                "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
                input_shape=tuple(x_train_processed.shape[1:]),
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
        self._model.fit(X, y, epochs=5)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def save(self):
        if self._model is not None:
            joblib.dump(self._model, self._model_path)
        else:
            raise TypeError("The model is not trained yet, use .train() before saving")

    def load(self):
        try:
            self._model = joblib.load(self._model_path)
        except:
            self._model = None
        return self


model_path = Path(__file__).parent / "model.joblib"
n_features = load_boston(return_X_y=True)[0].shape[1]
model = Model(model_path)


def get_model():
    return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train_processed = preprocessing.preprocess_mnist(x_train)
    model.train(x_train_processed, y_train)
    model.save()