import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from .data_prepare import rotating
from .data_prepare import crop_image
from .data_prepare import shear
from .data_prepare import crop_after_shearing
from .data_prepare import create_binary_masks
from .data_prepare import crop_image_neg

from dataset.data_config import HEIGHT
from dataset.data_config import WIDTH
from dataset.data_config import ROTATION_ANGLE_1
from dataset.data_config import SHEAR_FACTOR_X_1
from dataset.data_config import SHEAR_FACTOR_Y_1
from dataset.data_config import ROTATION_ANGLE_2
from dataset.data_config import SHEAR_FACTOR_X_2
from dataset.data_config import SHEAR_FACTOR_Y_2
from dataset.data_config import ROTATION_ANGLE_3
from dataset.data_config import ROTATION_ANGLE_4
from dataset.data_config import ROTATION_ANGLE_5
from dataset.data_config import ROTATION_ANGLE_6


def load_image(img):
    image = tf.convert_to_tensor(img, dtype=tf.float32)
    image = tf.cast(image, tf.float32)
    image = image/255.
    return image

def extract_frame_number(file_path):
    return int(file_path.split('(')[1].split(')')[0])

def read_dataset():
    path = './ds/images'
    img_names = [os.path.join(path, img) for img in os.listdir(path) if img.lower().endswith(('.jpg'))]
    masks_names = [os.path.join(path, mask) for mask in os.listdir(path) if mask.lower().endswith(('fuse.png'))]

    img_names = sorted(img_names, key=extract_frame_number)
    masks_names = sorted(masks_names, key=extract_frame_number)

    images = []
    maskes = []
    for img, mask in zip(img_names, masks_names):
        img = cv.imread(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        mask = cv.imread(mask)
        mask = cv.cvtColor(mask, cv.COLOR_RGB2BGR)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        if img is not None and mask is not None:
            resized_img = cv.resize(img, (WIDTH, HEIGHT), interpolation=cv.INTER_LINEAR) # 1080 / 4 i 1920 / 4
            resized_mask = cv.resize(mask, (WIDTH, HEIGHT), interpolation=cv.INTER_NEAREST)

            flipped_img_horizontal = cv.flip(resized_img, 1)
            flipped_mask_horizontal = cv.flip(resized_mask, 1)

            rotated_img = rotating(resized_img, ROTATION_ANGLE_1, typ='img')
            rotated_mask = rotating(resized_mask, ROTATION_ANGLE_1,typ='mask')
            rotated_img1 = rotating(resized_img, ROTATION_ANGLE_2, typ='img')
            rotated_mask1 = rotating(resized_mask, ROTATION_ANGLE_2,typ='mask')
            rotated_img2 = rotating(resized_img, ROTATION_ANGLE_3, typ='img')
            rotated_mask2 = rotating(resized_mask, ROTATION_ANGLE_3,typ='mask')

            cropped_rotated = crop_image(rotated_img, typ='img')
            cropped_rotated_mask = crop_image(rotated_mask, typ='mask')
            cropped_rotated1 = crop_image(rotated_img1, typ='img')
            cropped_rotated_mask1 = crop_image(rotated_mask1, typ='mask')
            cropped_rotated2 = crop_image(rotated_img2, typ='img')
            cropped_rotated_mask2 = crop_image(rotated_mask2, typ='mask')

            rotated_img = rotating(resized_img, ROTATION_ANGLE_4, typ='img')
            rotated_mask = rotating(resized_mask, ROTATION_ANGLE_4,typ='mask')
            rotated_img1 = rotating(resized_img, ROTATION_ANGLE_5, typ='img')
            rotated_mask1 = rotating(resized_mask, ROTATION_ANGLE_5,typ='mask')
            rotated_img2 = rotating(resized_img, ROTATION_ANGLE_6, typ='img')
            rotated_mask2 = rotating(resized_mask, ROTATION_ANGLE_6,typ='mask')

            cropped_rotated3 = crop_image_neg(rotated_img, typ='img')
            cropped_rotated_mask3 = crop_image_neg(rotated_mask, typ='mask')
            cropped_rotated4 = crop_image_neg(rotated_img1, typ='img')
            cropped_rotated_mask4 = crop_image_neg(rotated_mask1, typ='mask')
            cropped_rotated5 = crop_image_neg(rotated_img2, typ='img')
            cropped_rotated_mask5 = crop_image_neg(rotated_mask2, typ='mask')

            sheared_img = shear(resized_img, SHEAR_FACTOR_X_1, SHEAR_FACTOR_Y_1, typ='img')
            sheared_mask = shear(resized_mask, SHEAR_FACTOR_X_1, SHEAR_FACTOR_Y_1, typ='mask')
            sheared_img1 = shear(resized_img, SHEAR_FACTOR_X_2, SHEAR_FACTOR_Y_2, typ='img')
            sheared_mask1 = shear(resized_mask, SHEAR_FACTOR_X_2, SHEAR_FACTOR_Y_2, typ='mask')

            cropped_sheared = crop_after_shearing(sheared_img, typ='img')
            cropped_sheared_mask = crop_after_shearing(sheared_mask, typ='mask')
            cropped_sheared1 = crop_after_shearing(sheared_img1, typ='img')
            cropped_sheared_mask1 = crop_after_shearing(sheared_mask1, typ='mask')

            images.append(resized_img)
            images.append(flipped_img_horizontal)
            images.append(cropped_rotated)
            images.append(cropped_rotated1)
            images.append(cropped_rotated2)
            images.append(cropped_rotated3)
            images.append(cropped_rotated4)
            images.append(cropped_rotated5)
            images.append(cropped_sheared)
            images.append(cropped_sheared1)

            maskes.append(resized_mask)
            maskes.append(flipped_mask_horizontal)
            maskes.append(cropped_rotated_mask)
            maskes.append(cropped_rotated_mask1)
            maskes.append(cropped_rotated_mask2)
            maskes.append(cropped_rotated_mask3)
            maskes.append(cropped_rotated_mask4)
            maskes.append(cropped_rotated_mask5)
            maskes.append(cropped_sheared_mask)
            maskes.append(cropped_sheared_mask1)
        else:
            print("Error: Unable to load the image")

    images = np.array(images)
    maskes = np.array(maskes)

    binary_maskes = create_binary_masks(maskes)
    images = load_image(images)

    images = np.array(images)
    X_train, X_test, y_train, y_test = train_test_split(images, binary_maskes, test_size=0.10, random_state=42)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    return X_train, X_test, y_train, y_test


def get_train_data(train_x, train_y):
    train_x = tf.keras.utils.normalize(train_x, axis=1)
    xtrain, xvalid, ytrain, yvalid = train_test_split(train_x, train_y, test_size=0.1, random_state=42)

    return xtrain, xvalid, ytrain, yvalid