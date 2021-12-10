import tensorflow as tf
from keras.layers import Dense
import keras
from keras import Model


def get_example_learner(input_shape: tuple, output_shape: int) -> keras.Model:
    input = keras.Input(input_shape)
    dense = Dense(4, activation="relu")(input)
    output = Dense(output_shape, activation="sigmoid")(dense)
    model = Model(inputs=input, outputs=output, name="test")
    return model
