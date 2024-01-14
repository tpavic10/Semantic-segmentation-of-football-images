import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
import tensorflow as tf
from keras import losses
import numpy as np
from numpy import save
from sklearn.utils.class_weight import compute_class_weight

from .config import MODEL_BACKBONE
from .config import OPTIMIZER
from .config import LEARNING_RATE
from .config import EPOCHS
from .config import BATCH_SIZE
from .config import METRICS
from model.model_architecture import get_model 
from model.model_effitientNet import get_eff_model

def weighted_binary_crossentropy(y_true, y_pred, class_weights):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    y_true = tf.cast(y_true, tf.float32)
    class_weights = tf.constant(list(class_weights.values()), dtype=tf.float32)
    
    weighted_losses = -tf.reduce_sum(class_weights * y_true * tf.math.log(y_pred) +
                                     (1.0 - y_true) * tf.math.log(1.0 - y_pred), axis=-1)
    
    return tf.reduce_mean(weighted_losses)

def compute_weights(y_train):
    y = np.argmax(y_train, axis=-1)
    y = y.reshape(y.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    y = y.flatten()

    classes = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = classes,
                                        y = y                                                 
                                        )
    
    weights = dict(zip(classes, class_weights))
    return weights


def train_model(x_train, x_valid, y_train, y_valid, model_typ, weighted=True):

    if model_typ == 'effitientNet': 
        preprocess_input = sm.get_preprocessing(MODEL_BACKBONE)
        x_train = preprocess_input(x_train)
        x_valid = preprocess_input(x_valid)
        
        model = get_eff_model()

    else: 
        model = get_model()


    class_weights = {}
    if weighted:
        class_weights = compute_weights(y_train)
        # class_weights = {0: 3, 1: 1, 2: 8, 3: 1, 4: 1, 5: 1, 6: 1, 7: 4}
        model.compile(loss=lambda y_true, y_pred:weighted_binary_crossentropy(y_true, y_pred, class_weights), 
                    optimizer=OPTIMIZER(learning_rate=LEARNING_RATE), 
                    metrics=[METRICS])
        
        data = model.fit(x=x_train,
              y=y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_data=(x_valid, y_valid),
              class_weight=class_weights)
    
    else:
        model.compile(loss=losses.binary_crossentropy, 
                    optimizer=OPTIMIZER(learning_rate=LEARNING_RATE), 
                    metrics=[METRICS])
        
        data = model.fit(x=x_train,
              y=y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=1,
              validation_data=(x_valid, y_valid))

    
    if model_typ == 'effitientNet': 
        save('./effNet_result/train_loss_b3.npy', np.array(data.history['loss']))
        save('./effNet_result/train_iou_b3.npy', np.array(data.history['mean_io_u']))
        save('./effNet_result/val_loss_b3.npy', np.array(data.history['val_loss']))
        save('./effNet_result/val_iou_b3.npy', np.array(data.history['val_mean_io_u']))
        model.save_weights('./weights/model_effNetb3_1.h5')
    else:
        save('./custom_model_result/train_loss_custom.npy', np.array(data.history['loss']))
        save('./custom_model_result/train_iou_custom.npy', np.array(data.history['mean_io_u']))
        save('./custom_model_result/val_loss_custom.npy', np.array(data.history['val_loss']))
        save('./custom_model_result/val_iou_custom.npy', np.array(data.history['val_mean_io_u']))
        model.save_weights('./weights/custom_model_final.h5')

    return class_weights


