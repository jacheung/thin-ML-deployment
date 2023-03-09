import tensorflow as tf


def preprocess_mnist_tfds(image, label=None):
    # reshape and upsample to 3 channel for transfer learning models
    image = tf.image.grayscale_to_rgb(image)
    # normalize pixel values
    image = tf.cast(image, tf.float32) / 255.
    # resize with pad for mobilenetv2
    image = tf.image.resize_with_pad(image, target_height=224, target_width=224)
    return image, label


