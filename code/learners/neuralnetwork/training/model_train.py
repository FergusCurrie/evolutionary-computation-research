"""
Simple function for training a model. 

Author : Fergus Currie 
"""
import keras
import tensorflow as tf
import numpy as np
from typing import Callable


def train(
    model: keras.Model,
    X: np.array,
    y: np.array,
    optimiser: keras.optimizer_v2.adam.Adam,
    num_epochs: int,
    batch_size: int,
    loss_function: Callable,
) -> keras.Model:
    """
    Trains a keras model.

    Args:
        model (keras.Model): a keras model.
        X (np.array) : data in shape (#features, #instances)
        y (np.array) : class value in shape (#instances,)
        num_epochs (int): number of training epochs
        batch_size (int): number of instances in minibatch
        loss_function (function): a function which takes logits and returns a tf.float loss value

    Returns:
        keras.Model: the trained model
    """
    record = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    for epoch in range(num_epochs):
        # Sample a batch
        batch_mask = np.random.randint(0, len(X) - 1, batch_size)
        X_batch = X[batch_mask, :]
        y_batch = y[batch_mask]

        with tf.GradientTape() as tape:
            # Calculate gradient
            logits = model(X_batch)
            loss = loss_function(logits, y_batch)
            record["loss"].append(loss)

        # Apply gradients
        gradients = tape.gradient(loss, model.trainable_weights)
        optimiser.apply_gradients(zip(gradients, model.trainable_weights))
    return model
