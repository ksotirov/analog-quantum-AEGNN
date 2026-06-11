from enum import Enum


class Normalisation(Enum):
    NO_NORM = 'No Normalisation'
    SIMPLE_NORM = 'Simple Normalisation'
    SIMPLE_NORM_PER_LAYER = 'Simple Normalisation Per Layer'
    MIN_MAX_NORM = 'Min-Max Normalisation'
    MIN_MAX_NORM_PER_LAYER = 'Min-Max Normalisation Per Layer'