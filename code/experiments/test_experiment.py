import pandas as pd

from code.data_manager import get_cleveland, pandas_to_numpy_x_y
from code.neuralnetwork.training.model_train import train
from code.neuralnetwork.losses.loss_categorical_crossentropy import get_loss_categorical_crossentropy
from code.neuralnetwork.architechtures.architechture_factory import build_architechture
from code.neuralnetwork.architechtures.example_leaner import get_example_learner
import tensorflow as tf
import numpy as np
import keras

# model = keras.models.load_model('path/to/location')

EXPERIMENT_NAME = "test"


def run_experiment_test():

    number_members = 5
    for n in range(number_members):

        optimiser = tf.keras.optimizers.Adam()

        dataX, dataY = pandas_to_numpy_x_y(get_cleveland())

        model = build_architechture(
            input_shape=tuple([13]),
            num_outputs=1,
            get_architechture=get_example_learner,
        )

        trained_model = train(
            model=model,
            X=dataX,
            y=dataY,
            optimiser=optimiser,
            num_epochs=30,
            batch_size=128,
            loss_function=get_loss_categorical_crossentropy,
        )
        trained_model.save(f"data/models/{EXPERIMENT_NAME}_{n}.h5")

        print("done")
