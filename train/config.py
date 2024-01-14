from keras import metrics, optimizers

MODEL_BACKBONE = 'efficientnetb3'

OPTIMIZER = optimizers.Adam
LEARNING_RATE = 0.001
EPOCHS = 80
BATCH_SIZE = 2
METRICS = metrics.MeanIoU(num_classes=8)