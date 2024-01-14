import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

from .config import MODEL_BACKBONE
from .config import PRETRAINED_WEIGHTS


def get_eff_model():
    
    model = sm.Unet(MODEL_BACKBONE, encoder_weights=PRETRAINED_WEIGHTS, classes=8, activation="softmax")
    print(model.summary())

    return model