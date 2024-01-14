import cv2 as cv
import numpy as np

from dataset.data_config import HEIGHT
from dataset.data_config import WIDTH

def crop_image_neg(rgb_img, typ):
    if typ != 'mask':
        gray_img = cv.cvtColor(rgb_img.astype('uint8'), cv.COLOR_BGR2GRAY)
    else:
        gray_img = rgb_img

    crop_top_left = 0
    crop_bottom_right = 0
    crop_bottom_left = 0
    crop_top_right = 0
    for x in range (gray_img.shape[0]):
        if x == 0:
            for y in range (gray_img.shape[1]):
                if gray_img[x][y] == 0:
                    continue
                else:
                    crop_top_left = y
                    break
        elif x == gray_img.shape[0] - 1:
            for y in range (gray_img.shape[1] - 1, -1, -1):
                if gray_img[x][y] == 0:
                    continue
                else:
                    crop_bottom_right = y
                    break

    for x in range (gray_img.shape[1] - 1, -1, -1):
        if x == gray_img.shape[1] - 1:
            for y in range (gray_img.shape[0]):
                if gray_img[y][x] == 0:
                    continue
                else:
                    crop_top_right = y
                    break
        elif x == 0: 
            for y in range (gray_img.shape[0]- 1, -1, -1): # reverse loop
                if gray_img[y][x] == 0:
                    continue
                else:
                    crop_bottom_left = y
                    break

    crop_image = rgb_img[crop_top_right:crop_bottom_left, crop_top_left:crop_bottom_right]

    if typ != 'mask':
        resized_img = cv.resize(crop_image, (256, 256), interpolation=cv.INTER_LINEAR)
    else:
        resized_img = cv.resize(crop_image, (256, 256), interpolation=cv.INTER_NEAREST)
    return resized_img


def crop_image(rgb_img, typ):
    if typ != 'mask':
        gray_img = cv.cvtColor(rgb_img.astype('uint8'), cv.COLOR_BGR2GRAY)
    else:
        gray_img = rgb_img

    crop_top_left = 0
    crop_bottom_right = 0
    crop_bottom_left = 0
    crop_top_right = 0
    for y in range (gray_img.shape[1]):
        if y == 0:
            for x in range (gray_img.shape[0]):
                if gray_img[x][y] == 0:
                    continue
                else:
                    crop_top_left = x
                    break
        elif y == gray_img.shape[1] - 1:
            for x in range (gray_img.shape[0] - 1, -1, -1):
                if gray_img[x][y] == 0:
                    continue
                else:
                    crop_bottom_right = x
                    break

    for x in range (gray_img.shape[0]):
        if x == 0:
            for y in range (gray_img.shape[1] - 1, -1, -1):
                if gray_img[x][y] == 0:
                    continue
                else:
                    crop_top_right = y
                    break
        elif x == gray_img.shape[0] - 1: 
            for y in range (gray_img.shape[0]): 
                if gray_img[x][y] == 0:
                    continue
                else:
                    crop_bottom_left = y
                    break

    crop_image = rgb_img[crop_top_left:crop_bottom_right, crop_bottom_left:crop_top_right]

    if typ != 'mask':
        resized_img = cv.resize(crop_image, (WIDTH, HEIGHT), interpolation=cv.INTER_LINEAR)
    else:
        resized_img = cv.resize(crop_image, (WIDTH, HEIGHT), interpolation=cv.INTER_NEAREST)
    return resized_img

def crop_after_shearing(rgb_img, typ):
    if typ != 'mask':
        gray_img = cv.cvtColor(rgb_img.astype('uint8'), cv.COLOR_BGR2GRAY)
    else:
        gray_img = rgb_img

    crop_bottom_right = 0
    crop_top_right = 0
    for y in range (gray_img.shape[1] - 1, -1, -1):
        if y == gray_img.shape[1] - 1:
            for x in range (gray_img.shape[0]):
                if gray_img[x][y] == 0:
                    continue
                else:
                    crop_top_right = x
                    break
        else:
            break
    for y in range (gray_img.shape[0] - 1, -1, -1):
        if y == gray_img.shape[0] - 1:
            for x in range (gray_img.shape[1] - 1, -1, -1):
                if gray_img[y][x] == 0:
                    continue
                else:
                    crop_bottom_right = x
                    break
        else:
            break

    crop_image = rgb_img[crop_top_right:, :crop_bottom_right]

    if typ != 'mask':
        resized_img = cv.resize(crop_image, (WIDTH, HEIGHT), interpolation=cv.INTER_LINEAR)
    else:
        resized_img = cv.resize(crop_image, (WIDTH, HEIGHT), interpolation=cv.INTER_NEAREST)

    return resized_img

def rotating(img, rotation_angle, typ):
    center = (img.shape[1] // 2, img.shape[0] // 2) # center of the image
    rotation_matrix = cv.getRotationMatrix2D(center, rotation_angle, 1.0) # rotation matrix

    if typ != 'mask':
        rotated_img = cv.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
    else:
        rotated_img = cv.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]), flags=cv.INTER_NEAREST)
    

    return rotated_img

def shear(img, shear_factor_x, shear_factor_y, typ):
    shear_matrix = np.array([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]], dtype=np.float32) # shear matrix
    if typ != 'mask':
        sheared_img = cv.warpAffine(img, shear_matrix, (img.shape[1], img.shape[0]))
    else:
        sheared_img = cv.warpAffine(img, shear_matrix, (img.shape[1], img.shape[0]), flags=cv.INTER_NEAREST)

    return sheared_img


def create_binary_masks(maskes):
    mask_binary = np.zeros(shape=(maskes.shape[0], 256, 256, 8), dtype=int)
    pixel_values = [38, 90, 101, 116, 123, 127, 167, 179]

    for value, ch in zip(pixel_values, [0,1,2,3,4,5,6,7]):
        mask = (maskes == value).astype(int)
        
        # mask to the corresponding channel in the final array
        mask_binary[..., ch] = mask

    return mask_binary
        