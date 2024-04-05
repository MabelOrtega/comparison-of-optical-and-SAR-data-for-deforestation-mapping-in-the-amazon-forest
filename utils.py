# Import libraries
import os
import sys
import time
import math
import numpy as np
from PIL import Image
import tensorflow as tf
from libtiff import TIFF
import skimage.morphology
import matplotlib.pyplot as plt
from skimage.filters import rank
from sklearn.utils import shuffle
from skimage.morphology import disk
from skimage.transform import resize
import tensorflow.keras.backend as K
from contextlib import redirect_stdout
from sklearn.metrics import confusion_matrix
from skimage.util.shape import view_as_windows
from sklearn.metrics import average_precision_score
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Model, load_model, Sequential
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Function to load yaml configuration file
def load_config(CONFIG_PATH, config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

def load_tif_image(patch):
    # Read tiff Image
    print(patch)
    img_tif = TIFF.open(patch)
    img = img_tif.read_image()
    return img


def load_SAR_image(patch):
    '''Function to read SAR images'''
    print(patch)
    img_tif = TIFF.open(patch)
    db_img = img_tif.read_image()
    temp_db_img = 10 ** (db_img / 10)
    temp_db_img[temp_db_img > 1] = 1
    return temp_db_img


def resize_image(image, height, width):
    im_resized = np.zeros((height, width, image.shape[2]), dtype='float32')
    for b in range(image.shape[2]):
        band = Image.fromarray(image[:, :, b])
        # (width, height) = (ref_2019.shape[1], ref_2019.shape[0])
        im_resized[:, :, b] = np.array(band.resize((width, height), resample=Image.NEAREST))
    return im_resized


def filter_outliers(img, bins=1000000, bth=0.001, uth=0.999, mask=[0]):
    img[np.isnan(img)] = 0  # Filter NaN values.
    if len(mask) == 1:
        mask = np.zeros((img.shape[:2]), dtype='int64')
    for band in range(img.shape[-1]):
        hist = np.histogram(img[:mask.shape[0], :mask.shape[1]][mask != 2, band].ravel(),
                            bins=bins)  # select not testing pixels
        cum_hist = np.cumsum(hist[0]) / hist[0].sum()
        max_value = np.ceil(100 * hist[1][len(cum_hist[cum_hist < uth])]) / 100
        min_value = np.ceil(100 * hist[1][len(cum_hist[cum_hist < bth])]) / 100
        img[:, :, band][img[:, :, band] > max_value] = max_value
        img[:, :, band][img[:, :, band] < min_value] = min_value
    return img


def binary_mask_cloud(image, th_up):
    cloud_mask = image.copy()
    cloud_mask[cloud_mask >= th_up] = -1
    cloud_mask[cloud_mask > 0] = 0
    cloud_mask[cloud_mask == -1] = 1
    cloud_mask = np.squeeze(cloud_mask)
    print(np.unique(cloud_mask))
    return cloud_mask


def mask_no_considered(image_ref, buffer, past_ref):
    # Creation of buffer for pixel no considered
    image_ref_ = image_ref.copy()
    im_dilate = skimage.morphology.dilation(image_ref_, disk(buffer))
    im_erosion = skimage.morphology.erosion(image_ref_, disk(buffer))
    inner_buffer = image_ref_ - im_erosion
    inner_buffer[inner_buffer == 1] = 2
    outer_buffer = im_dilate - image_ref_
    outer_buffer[outer_buffer == 1] = 2

    # 1 deforestation, 2 unknown
    image_ref_[outer_buffer + inner_buffer == 2] = 2
    image_ref_[past_ref == 1] = 2
    return image_ref_


def normalization(image, norm_type=1):
    image_reshaped = image.reshape((image.shape[0] * image.shape[1]), image.shape[2])
    if (norm_type == 1):
        scaler = StandardScaler()
    if (norm_type == 2):
        scaler = MinMaxScaler(feature_range=(0, 1))
    if (norm_type == 3):
        scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(image_reshaped)
    image_normalized = scaler.fit_transform(image_reshaped)
    image_normalized1 = image_normalized.reshape(image.shape[0], image.shape[1], image.shape[2])
    return image_normalized1


def create_mask(size_rows, size_cols, grid_size=(6, 3)):
    num_tiles_rows = size_rows // grid_size[0]
    num_tiles_cols = size_cols // grid_size[1]
    print('Tiles size: ', num_tiles_rows, num_tiles_cols)
    patch = np.ones((num_tiles_rows, num_tiles_cols))
    mask = np.zeros((num_tiles_rows * grid_size[0], num_tiles_cols * grid_size[1]))
    count = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            count = count + 1
            mask[num_tiles_rows * i:(num_tiles_rows * i + num_tiles_rows),
            num_tiles_cols * j:(num_tiles_cols * j + num_tiles_cols)] = patch * count
    # plt.imshow(mask)
    print('Mask size: ', mask.shape)
    return mask


def extract_patches_mask_indices(input_image, patch_size, stride):
    h, w = input_image.shape
    image_indices = np.arange(h * w).reshape(h, w)
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(image_indices, window_shape_array, step=stride))
    num_row, num_col, row, col = patches_array.shape
    patches_array = patches_array.reshape(num_row * num_col, row, col)
    return patches_array


def create_idx_image(ref_mask):
    im_idx = np.arange(ref_mask.shape[0] * ref_mask.shape[1]).reshape(ref_mask.shape[0], ref_mask.shape[1])
    return im_idx


def extract_patches(im_idx, patch_size, overlap):
    '''overlap range: 0 - 1 '''
    row_steps, cols_steps = int((1 - overlap) * patch_size[0]), int((1 - overlap) * patch_size[1])
    patches = skimage.util.view_as_windows(im_idx, patch_size, step=(row_steps, cols_steps))
    return patches


def retrieve_idx_percentage(reference, patches_idx_set, pertentage=5):
    count = 0
    new_idx_patches = []
    reference_vec = reference.reshape(reference.shape[0] * reference.shape[1])
    for patchs_idx in patches_idx_set:
        patch_ref = reference_vec[patchs_idx]
        class1 = patch_ref[patch_ref == 1]
        if len(class1) >= int((patch_size ** 2) * (pertentage / 100)):
            count = count + 1
            new_idx_patches.append(patchs_idx)
    return np.asarray(new_idx_patches)


def accuracy_cpu(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() > 0.5
    y_true = y_true.ravel()
    pred = pred[y_true != 2]
    y_true = y_true[y_true != 2]
    return np.mean(pred == y_true)


def euclidean_distance(x, y):
    # print(x.get_shape(),y.get_shape() )
    distance = tf.sqrt(tf.reduce_sum(tf.square(x - y), -1, keepdims=True))  # ok
    # distance = tf.math.l2_normalize(distance1, axis=-1, epsilon=1e-12)
    return distance


def euclidean_distance_np(x, y):
    dist = np.sqrt(np.sum(np.square(x - y), axis=-1, keepdims=True))
    # dist = np.linalg.norm(dist1, axis = -1, keepdims = True)
    # print(dist.shape)
    return dist


def eucl_dist_output_shape(shapes):
    # print(shape1.get_shape())
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred, weights):
    weights = K.variable(weights)
    margin = 2

    # Flatten y_true and y_pred
    flatten_y_pred = K.reshape(y_pred, [-1])
    flatten_y_true = K.reshape(y_true, [-1])

    # Supposed 2 is the ignored label
    pos_considered = tf.math.not_equal(flatten_y_true, 2)
    indices = tf.where(pos_considered)
    filtered_y_pred = tf.gather(flatten_y_pred, indices)
    filtered_y_true = tf.gather(flatten_y_true, indices)

    # Computing loss
    square_pred = K.square(filtered_y_pred)
    margin_square = K.square(K.maximum(margin - filtered_y_pred, 0))
    loss = (1 - filtered_y_true) * square_pred * weights[0] + filtered_y_true * margin_square * weights[1]

    return tf.reduce_mean(loss)

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred)
        loss = loss * weights
        loss = - K.mean(loss, -1)
        return loss

    return loss

def matrics_AA_recall(thresholds_, prob_map, ref_reconstructed, mask_amazon_ts_, px_area):
    thresholds = thresholds_
    metrics_all = []

    for thr in thresholds:
        print(thr)

        img_reconstructed = np.zeros_like(prob_map)
        img_reconstructed[prob_map >= thr] = 1

        mask_areas_pred = np.ones_like(ref_reconstructed)
        area = skimage.morphology.area_opening(img_reconstructed, area_threshold=px_area, connectivity=1)
        area_no_consider = img_reconstructed - area
        mask_areas_pred[area_no_consider == 1] = 0

        # Mask areas no considered reference
        mask_borders = np.ones_like(img_reconstructed)
        # ref_no_consid = np.zeros((ref_reconstructed.shape))
        mask_borders[ref_reconstructed == 2] = 0

        mask_no_consider = mask_areas_pred * mask_borders
        ref_consider = mask_no_consider * ref_reconstructed
        pred_consider = mask_no_consider * img_reconstructed

        ref_final = ref_consider[mask_amazon_ts_ == 1]
        pre_final = pred_consider[mask_amazon_ts_ == 1]

        # Metrics
        cm = confusion_matrix(ref_final, pre_final)
        # TN = cm[0,0]
        FN = cm[1, 0]
        TP = cm[1, 1]
        FP = cm[0, 1]
        precision_ = TP / (TP + FP)
        recall_ = TP / (TP + FN)
        aa = (TP + FP) / len(ref_final)
        mm = np.hstack((recall_, precision_, aa))
        metrics_all.append(mm)
    metrics_ = np.asarray(metrics_all)
    return metrics_


def pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_x, patch_size_y, patches_pred):
    count = 0
    img_reconstructed = np.zeros((h, w)).astype(np.float32)
    for i in range(0, num_patches_y):
        for j in range(0, num_patches_x):
            img_reconstructed[patch_size_x * j:patch_size_x * (j + 1), patch_size_y * i:patch_size_y * (i + 1)] = \
            patches_pred[count]
            count += 1
    return img_reconstructed


def complete_nan_values(metrics):
    vec_prec = metrics[:, 1]
    for j in reversed(range(len(vec_prec))):
        if np.isnan(vec_prec[j]):
            vec_prec[j] = 2 * vec_prec[j + 1] - vec_prec[j + 2]
            if vec_prec[j] >= 1:
                vec_prec[j] = 1
    metrics[:, 1] = vec_prec
    return metrics


def batch_generator(batches, image, reference, target_size, number_class):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    image = image.reshape(-1, image.shape[-1])
    reference = reference.reshape(image.shape[0] * image.shape[1])
    while True:
        batch_x, batch_y = next(batches)
        batch_x = np.squeeze(batch_x.astype('int64'))
        # print(batch_x.shape)
        batch_img = np.zeros((batch_x.shape[0], target_size, target_size, image.shape[-1]))
        batch_ref = np.zeros((batch_x.shape[0], target_size, target_size, number_class))

        for i in range(batch_x.shape[0]):
            if np.random.rand() > 0.5:
                batch_x[i] = np.rot90(batch_x[i], 1)
            batch_img[i] = image[batch_x[i]]
            batch_ref[i] = tf.keras.utils.to_categorical(reference[batch_x[i]], number_class)

        yield (batch_img, batch_ref)


class ImageLoader:
    def __init__(self, img_type):
        self.img_type = img_type

    def load_images(self):
        if self.img_type == 'L8':
            return self.load_landsat_images()
        elif self.img_type == 'S2':
            return self.load_sentinel2_images()
        elif self.img_type == 'S1':
            return self.load_sentinel1_images()

    def load_landsat_images(self):
        # Load images
        ref_2019 = load_tif_image(root_path + 'Images/Landsat8/r10m_def_2019.tif').astype('float32')
        opt_2018 = load_tif_image(root_path + 'Images/Landsat8/cut_land8_2018.tif').astype('float32')
        opt_2019 = load_tif_image(root_path + 'Images/Landsat8/cut_land8_2019.tif').astype('float32')

        # Resize images
        opt_2018 = resize_image(opt_2018.copy(), ref_2019.shape[0], ref_2019.shape[1])
        opt_2019 = resize_image(opt_2019.copy(), ref_2019.shape[0], ref_2019.shape[1])

        # Filter outliers
        opt_2018 = self.filter_outliers(opt_2018.copy())
        opt_2019 = self.filter_outliers(opt_2019.copy())

        image_stack = np.concatenate((opt_2018, opt_2019), axis=-1)
        print('landsat_resize:', image_stack.shape)
        return image_stack

    def load_sentinel2_images(self):
        # Load images
        sent2_2018_1 = load_tif_image(root_path + 'Images/Sentinel2/2018_10m_b2348.tif').astype('float32')
        sent2_2018_2 = load_tif_image(root_path + 'Images/Sentinel2/2018_20m_b5678a1112.tif').astype('float32')

        # Resize bands of 20m
        sent2_2018_2 = resize_image(sent2_2018_2.copy(), sent2_2018_1.shape[0], sent2_2018_1.shape[1])
        sent2_2018 = np.concatenate((sent2_2018_1, sent2_2018_2), axis=-1)

        image_stack = np.concatenate((sent2_2018, sent2_2019), axis=-1)
        print('Image stack:', image_stack.shape)
        return image_stack

    def load_sentinel1_images(self):
        # Load Sentinel-1 images
        sar_2018 = np.expand_dims(
            load_SAR_image(root_path + 'Images/Sentinel1/cut_sent1_2018.tif').astype('float32'), axis=-1)
        sar_2019 = np.expand_dims(
            load_SAR_image(root_path + 'Images/Sentinel1/cut_sent1_2019.tif').astype('float32'), axis=-1)

        # Filter outliers
        sar_2018 = self.filter_outliers(sar_2018.copy())
        sar_2019 = self.filter_outliers(sar_2019.copy())
        image_stack = np.concatenate((sar_2018, sar_2019), axis=-1)
        print('Image stack:', image_stack.shape)
        return image_stack

    def filter_outliers(self, img):
        # Filter outliers
        return filter_outliers(img)

