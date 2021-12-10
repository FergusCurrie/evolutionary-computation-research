import tensorflow as tf
import numpy as np


def get_loss_categorical_crossentropy(logits: tf.Tensor, y_true: np.array) -> tf.Tensor:
    """
    Standard categorical cross entropy. For now we treat logits as (m,n) and true as (m,)

    Args:
        logits (tf.variable): raw predictions from model, shape : (batchsize, #outputs)
        y_true (np.array): true data, shape : (batchsize, )

    Returns:
        [type]: [description]
    """
    dim = tf.shape(logits).numpy()[1]
    y_batch = np.repeat(y_true[..., np.newaxis], dim, axis=1)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    return cce(logits, y_batch)
