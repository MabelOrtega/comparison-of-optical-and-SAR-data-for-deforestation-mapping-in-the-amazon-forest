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
#from osgeo import ogr, gdal
import matplotlib.pyplot as plt
from skimage.filters import rank
from sklearn.utils import shuffle
from skimage.morphology import disk
from skimage.transform import resize
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from contextlib import redirect_stdout
from sklearn.metrics import confusion_matrix
from skimage.util.shape import view_as_windows
from sklearn.metrics import average_precision_score
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Model, load_model, Sequential
#from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]=""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Functions
def load_tif_image(patch):
    # Read tiff Image
    print (patch)
    img_tif = TIFF.open(patch)
    img = img_tif.read_image()
    return img

def load_SAR_image(patch):
    '''Function to read SAR images'''
    print (patch)
    img_tif = TIFF.open(patch)
    db_img = img_tif.read_image()
    temp_db_img = 10**(db_img/10)
    temp_db_img[temp_db_img>1] = 1
    return temp_db_img

def resize_image(image, height, width):
    im_resized = np.zeros((height, width, image.shape[2]), dtype='float32')
    for b in range(image.shape[2]):
        band = Image.fromarray(image[:,:,b])
        #(width, height) = (ref_2019.shape[1], ref_2019.shape[0])
        im_resized[:,:,b] = np.array(band.resize((width, height), resample=Image.NEAREST))
    return im_resized

def filter_outliers(img, bins=1000000, bth=0.001, uth=0.999, mask=[0]):
    img[np.isnan(img)]=0 # Filter NaN values.
    if len(mask)==1:
        mask = np.zeros((img.shape[:2]), dtype='int64')
    for band in range(img.shape[-1]):
        hist = np.histogram(img[:mask.shape[0], :mask.shape[1]][mask!=2, band].ravel(),bins=bins) # select not testing pixels
        cum_hist = np.cumsum(hist[0])/hist[0].sum()
        max_value = np.ceil(100*hist[1][len(cum_hist[cum_hist<uth])])/100
        min_value = np.ceil(100*hist[1][len(cum_hist[cum_hist<bth])])/100
        img[:,:, band][img[:,:, band]>max_value] = max_value
        img[:,:, band][img[:,:, band]<min_value] = min_value
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
    outer_buffer = im_dilate-image_ref_
    outer_buffer[outer_buffer == 1] = 2
    
    # 1 deforestation, 2 unknown
    image_ref_[outer_buffer + inner_buffer == 2 ] = 2
    image_ref_[past_ref == 1] = 2
    return image_ref_
    
def normalization(image, norm_type = 1):
    image_reshaped = image.reshape((image.shape[0]*image.shape[1]),image.shape[2])
    if (norm_type == 1):
        scaler = StandardScaler()
    if (norm_type == 2):
        scaler = MinMaxScaler(feature_range=(0,1))
    if (norm_type == 3):
        scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(image_reshaped)
    image_normalized = scaler.fit_transform(image_reshaped)
    image_normalized1 = image_normalized.reshape(image.shape[0],image.shape[1],image.shape[2])
    return image_normalized1
    
def create_mask(size_rows, size_cols, grid_size=(6,3)):
    num_tiles_rows = size_rows//grid_size[0]
    num_tiles_cols = size_cols//grid_size[1]
    print('Tiles size: ', num_tiles_rows, num_tiles_cols)
    patch = np.ones((num_tiles_rows, num_tiles_cols))
    mask = np.zeros((num_tiles_rows*grid_size[0], num_tiles_cols*grid_size[1]))
    count = 0
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            count = count+1
            mask[num_tiles_rows*i:(num_tiles_rows*i+num_tiles_rows), num_tiles_cols*j:(num_tiles_cols*j+num_tiles_cols)] = patch*count
    #plt.imshow(mask)
    print('Mask size: ', mask.shape)
    return mask

def extract_patches_mask_indices(input_image, patch_size, stride):
    h, w = input_image.shape
    image_indices = np.arange(h*w).reshape(h,w)
    window_shape = patch_size
    window_shape_array = (window_shape, window_shape)
    patches_array = np.array(view_as_windows(image_indices, window_shape_array, step = stride))    
    num_row,num_col,row,col = patches_array.shape
    patches_array = patches_array.reshape(num_row*num_col,row,col)
    return patches_array

def create_idx_image(ref_mask):
    im_idx = np.arange(ref_mask.shape[0] * ref_mask.shape[1]).reshape(ref_mask.shape[0] , ref_mask.shape[1])
    return im_idx

def extract_patches(im_idx, patch_size, overlap):
    '''overlap range: 0 - 1 '''
    row_steps, cols_steps = int((1-overlap) * patch_size[0]), int((1-overlap) * patch_size[1])
    patches = skimage.util.view_as_windows(im_idx, patch_size, step=(row_steps, cols_steps))
    return patches
    
def retrieve_idx_percentage(reference, patches_idx_set, pertentage = 5):
    count = 0
    new_idx_patches = []
    reference_vec = reference.reshape(reference.shape[0]*reference.shape[1])
    for patchs_idx in patches_idx_set:
        patch_ref = reference_vec[patchs_idx]
        class1 = patch_ref[patch_ref==1]
        if len(class1) >= int((patch_size**2)*(pertentage/100)):
            count = count + 1
            new_idx_patches.append(patchs_idx)
    return np.asarray(new_idx_patches)

def pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_x, patch_size_y, patches_pred):
    count = 0
    img_reconstructed = np.zeros((h, w)).astype(np.float32)
    for i in range(0,num_patches_y):
        for j in range(0,num_patches_x):
            img_reconstructed[patch_size_x*j:patch_size_x*(j+1),patch_size_y*i:patch_size_y*(i+1)]=patches_pred[count]
            count+=1
    return img_reconstructed
    
def matrics_AA_recall(thresholds_, prob_map, ref_reconstructed, mask_amazon_ts_, px_area):
    thresholds = thresholds_    
    metrics_all = []
    
    for thr in thresholds:
        print(thr)  

        img_reconstructed = np.zeros_like(prob_map).astype(np.int8)
        img_reconstructed[prob_map >= thr] = 1
    
        mask_areas_pred = np.ones_like(ref_reconstructed)
        area = skimage.morphology.area_opening(img_reconstructed, area_threshold = px_area, connectivity=1)
        area_no_consider = img_reconstructed-area
        mask_areas_pred[area_no_consider==1] = 0
        
        # Mask areas no considered reference
        mask_borders = np.ones_like(img_reconstructed)
        #ref_no_consid = np.zeros((ref_reconstructed.shape))
        mask_borders[ref_reconstructed==2] = 0
        #mask_borders[ref_reconstructed==-1] = 0
        
        mask_no_consider = mask_areas_pred * mask_borders 
        ref_consider = mask_no_consider * ref_reconstructed
        pred_consider = mask_no_consider*img_reconstructed
        
        ref_final = ref_consider[mask_amazon_ts_==1]
        pre_final = pred_consider[mask_amazon_ts_==1]
        
        # Metrics
        cm = confusion_matrix(ref_final, pre_final)
        #TN = cm[0,0]
        FN = cm[1,0]
        TP = cm[1,1]
        FP = cm[0,1]
        precision_ = TP/(TP+FP)
        recall_ = TP/(TP+FN)
        aa = (TP+FP)/len(ref_final)
        mm = np.hstack((recall_, precision_, aa))
        metrics_all.append(mm)
    metrics_ = np.asarray(metrics_all)
    return metrics_
    
def complete_nan_values(metrics):
    vec_prec = metrics[:,1]
    for j in reversed(range(len(vec_prec))):
        if np.isnan(vec_prec[j]):
            vec_prec[j] = 2*vec_prec[j+1]-vec_prec[j+2]
            if vec_prec[j] >= 1:
                vec_prec[j] == 1
    metrics[:,1] = vec_prec
    return metrics 

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
            loss = y_true * K.log(y_pred) + (1-y_true) * K.log(1-y_pred)
            loss = loss * weights 
            loss = - K.mean(loss, -1)
            return loss
        return loss

# U-Net model
def build_unet(input_shape, nb_filters, n_classes):
    input_img = Input(input_shape)
 
    conv1 = Conv2D(nb_filters[0], (3 , 3) , activation='relu' , padding='same', name = 'conv1')(input_img) 
    pool1 = MaxPool2D((2 , 2))(conv1)
  
    conv2 = Conv2D(nb_filters[1], (3 , 3) , activation='relu' , padding='same', name = 'conv2')(pool1)
    pool2 = MaxPool2D((2 , 2))(conv2)
  
    conv3 = Conv2D(nb_filters[2], (3 , 3) , activation='relu' , padding='same', name = 'conv3')(pool2)
    pool3 = MaxPool2D((2 , 2))(conv3)
  
    conv4 = Conv2D(nb_filters[2], (3 , 3) , activation='relu' , padding='same', name = 'conv4')(pool3)
    
    conv5 = Conv2D(nb_filters[2], (3 , 3) , activation='relu' , padding='same', name = 'conv5')(conv4)
    
    conv6 = Conv2D(nb_filters[2], (3 , 3) , activation='relu' , padding='same', name = 'conv6')(conv5)
     
    tconv3 = Conv2D(nb_filters[2], (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling3')(UpSampling2D(size = (2,2))(conv6))
    merged3 = concatenate([conv3, tconv3], name='concatenate3')
      
    tconv2 = Conv2D(nb_filters[1], (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling2')(UpSampling2D(size = (2,2))(merged3))
    merged2 = concatenate([conv2, tconv2], name='concatenate2')
  
    tconv1 = Conv2D(nb_filters[0], (3 , 3), activation = 'relu', padding = 'same', name = 'upsampling1')(UpSampling2D(size = (2,2))(merged2))
    merged1 = concatenate([conv1, tconv1], name='concatenate1')
        
    output = Conv2D(n_classes,(1,1), activation = 'softmax')(merged1)
    
    return Model(input_img , output)

# Residual block 
def resnet_block(x, n_filter, ind):
    x_init = x
    ## Conv 1
    x =  Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res1_net'+str(ind))(x)
    x = Dropout(0.5, name = 'drop_net'+str(ind))(x)
    ## Shortcut
    s  = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name = 'res2_net'+str(ind))(x_init)
    ## Add
    x = Add()([x, s])
    return x

# Residual U-Net model
def build_resunet(input_shape, nb_filters, n_classes):
    '''Base network to be shared (eq. to feature extraction)'''
    input = Input(shape = input_shape, name="input_enc_net")
    
    res_block1 = resnet_block(input, nb_filters[0], 1)
    pool1 = MaxPool2D((2 , 2), name='pool_net1')(res_block1)
    
    res_block2 = resnet_block(pool1, nb_filters[1], 2)
    pool2 = MaxPool2D((2 , 2), name='pool_net2')(res_block2)
    
    res_block3 = resnet_block(pool2, nb_filters[2], 3)
    pool3 = MaxPool2D((2 , 2), name='pool_net3')(res_block3)
    
    res_block4 = resnet_block(pool3, nb_filters[2], 4)
    
    res_block5 = resnet_block(res_block4, nb_filters[2], 5)
    
    res_block6 = resnet_block(res_block5, nb_filters[2], 6)
    
    upsample3 = Conv2D(nb_filters[2], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net3')(UpSampling2D(size = (2,2))(res_block6))
    
    merged3 = concatenate([res_block3, upsample3], name='concatenate3')

    upsample2 = Conv2D(nb_filters[1], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net2')(UpSampling2D(size = (2,2))(merged3))
                                                 
    merged2 = concatenate([res_block2, upsample2], name='concatenate2')
                                                                                          
    upsample1 = Conv2D(nb_filters[0], (3 , 3), activation = 'relu', padding = 'same', 
                       name = 'upsampling_net1')(UpSampling2D(size = (2,2))(merged2))
    merged1 = concatenate([res_block1, upsample1], name='concatenate1')

    output = Conv2D(n_classes,(1,1), activation = 'softmax', padding = 'same', name = 'output')(merged1)
                                                                                                           
    return Model(input, output)
    