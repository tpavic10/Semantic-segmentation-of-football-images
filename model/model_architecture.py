from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPool2D
from keras.layers import Concatenate
from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.models import Model


def encoder(X, filters, rate=0.2):

    x = BatchNormalization()(X)
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Dropout(rate)(x)
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    
    return x

def decoder(x, skip_x, filters, rate):
    
    # skip_x is a skip connection to high-resolution features from encoder
    y = BatchNormalization()(x)
    y = Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same', activation='relu')(y)
    y = encoder(Concatenate()([y, skip_x]), filters, rate)
    return y


def get_model():
    InputL = Input(shape=(256, 256, 3), name="InputImage")

    # ENCODER - remember outpust before pooling for skip-connections in UNet
    enc1 = encoder(InputL, filters=64, rate=0.1)
    x1 = MaxPool2D()(enc1)
    enc2 = encoder(x1, filters=128, rate=0.1)
    x2 = MaxPool2D()(enc2)
    enc3 = encoder(x2, filters=256, rate=0.2)
    x3 = MaxPool2D()(enc3)
    enc4 = encoder(x3, filters=512, rate=0.2)
    x4 = MaxPool2D()(enc4)

    # Encoding Layer
    x = BatchNormalization()(x4)
    x = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    x = Dropout(0.3)(x)
    encodings = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu')(x)

    # DECODER
    d = decoder(encodings, enc4, filters=512, rate=0.2)
    d = decoder(d, enc3, filters=256, rate=0.2)
    d = decoder(d, enc2, filters=128, rate=0.1)
    d = decoder(d, enc1, filters=64, rate=0.1)

    # Output
    conv_out = Conv2D(8, kernel_size=3, padding='same', activation='softmax', name="Segmentation_layer")(d)

    model = Model(InputL, conv_out, name="UNet_model")
    return model