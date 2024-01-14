import segmentation_models as sm
import tensorflow as tf

from model.model_architecture import get_model
from model.model_effitientNet import get_eff_model
from .config import MODEL_BACKBONE
from .config import LEARNING_RATE
from .config import OPTIMIZER
from .config import METRICS
from .config import LOSS

def weighted_binary_crossentropy(y_true, y_pred, class_weights):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    y_true = tf.cast(y_true, tf.float32)
    class_weights = tf.constant(list(class_weights.values()), dtype=tf.float32)
    
    weighted_losses = -tf.reduce_sum(class_weights * y_true * tf.math.log(y_pred) +
                                     (1.0 - y_true) * tf.math.log(1.0 - y_pred), axis=-1)
    
    return tf.reduce_mean(weighted_losses)


def test_model(x_test, y_test, model_typ='custom', class_weights={}):
    
    if model_typ == 'effitientNet': 
        preprocess_input = sm.get_preprocessing(MODEL_BACKBONE)
        x_train = preprocess_input(x_train)
        x_valid = preprocess_input(x_valid)
        
        model = get_eff_model()
        model.load_weights('./weights/model_effNetb3_1.h5')

    else: 
        model = get_model()
        model.load_weights('./weights/custom_model.h5')

    model.compile(loss=LOSS,
                optimizer=OPTIMIZER(learning_rate=LEARNING_RATE), 
                metrics=[METRICS])


    x_test = tf.keras.utils.normalize(x_test, axis=1)
    report = model.evaluate(x_test, y_test, verbose=1)
    print(f"Report after testing: {report}")