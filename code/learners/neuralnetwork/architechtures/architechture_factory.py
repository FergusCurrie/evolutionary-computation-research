from typing import Callable
import keras
from keras import Model


def build_architechture(input_shape: tuple, num_outputs: int, get_architechture: Callable) -> keras.Model:
    return get_architechture(input_shape, num_outputs)
