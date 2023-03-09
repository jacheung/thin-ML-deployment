import numpy as np
import tensorflow as tf


def expand_greyscale_image_channels(grey_pil_image):
    grey_image_arr = np.array(grey_pil_image)
    grey_image_arr = np.expand_dims(grey_image_arr, -1)
    grey_image_arr_3_channel = grey_image_arr.repeat(3, axis=-1)
    return grey_image_arr_3_channel


def preprocess_mnist(images: np.array):
    # reshape and upsample to 3 channel for transfer learning models
    images = expand_greyscale_image_channels(images)
    # normalize pixel values
    images = images.astype('float32') / 255.0
    # resize with pad for mobilenetv2
    images = tf.image.resize_with_pad(images, target_height=224, target_width=224)
    return images


def preprocess_mnist_tfds(image, label):
    # reshape and upsample to 3 channel for transfer learning models
    image = tf.image.grayscale_to_rgb(image)
    # normalize pixel values
    image = tf.cast(image, tf.float32) / 255.
    # resize with pad for mobilenetv2
    image = tf.image.resize_with_pad(image, target_height=224, target_width=224)
    return image, label


