import tensorflow as tf
from keras import metrics, optimizers, losses

MODEL_BACKBONE = 'mobilenet'

OPTIMIZER = optimizers.Adam
LEARNING_RATE = 0.001
METRICS = metrics.MeanIoU(num_classes=8)
LOSS = losses.binary_crossentropy

PIXEL_MAPP = {38: (255, 0, 29),
              90: (27, 71, 151),
              101: (201, 19, 223),
              116: (111, 48, 253),
              123: (255, 160, 1),
              127: (137, 126, 126),
              167: (254, 233, 3),
              179: (238, 171, 171)}