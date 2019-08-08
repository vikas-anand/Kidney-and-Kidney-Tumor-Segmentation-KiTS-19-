import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os
import re
import glob
import pandas as pd
import random
import imgaug
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean
from os import path, mkdir
import tensorflow as tf 
import numpy as np 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import metrics
from tensorflow.keras.utils import*
from tensorflow.keras import backend as K

def imshow(*args,**kwargs):
    """ Handy function to show multiple plots in on row, possibly with different cmaps and titles
    Usage:
    imshow(img1, title="myPlot")
    imshow(img1,img2, title=['title1','title2'])
    imshow(img1,img2, cmap='hot')
    imshow(img1,img2,cmap=['gray','Blues']) """
    cmap = kwargs.get('cmap', 'gray')
    title= kwargs.get('title','')
    axis_off = kwargs.get('axis_off','')
    if len(args)==0:
        raise ValueError("No images given to imshow")
    elif len(args)==1:
        plt.title(title)
        plt.imshow(args[0], interpolation='none')
    else:
        n=len(args)
        if type(cmap)==str:
            cmap = [cmap]*n
        if type(title)==str:
            title= [title]*n
        x = 1
        y = np.ceil(n/x)
        plt.figure(figsize=(3*y, 3.2*x))
        # plt.figure()
        for i in range(n):
            plt.subplot(x,y,i+1)
            plt.title(title[i])
            plt.imshow(args[i], cmap[i])
            if axis_off:
                plt.axis('off')

    # plt.show()


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,image_path, mask_path, data_dir, image_size=(512,512), target_image_size=(256,256), 
                  batch_size=32, n_classes=3, n_channels=1,
                  shuffle=True, transform=None):
        'Initialization'
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.data_dir = data_dir

        self.images = sorted(image_path + i for i in pd.read_csv(data_dir)['Image_path'])
        self.masks = sorted(mask_path + i for i in pd.read_csv(data_dir)['Image_path'])

        self.image_size = image_size
        self.target_image_size = target_image_size
        self.n_channels = n_channels
       
        self.transform = transform
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        image_batch_list = self.images[index*self.batch_size:(index+1)*self.batch_size]
        mask_batch_list = self.masks[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(image_batch_list, mask_batch_list)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            mapIndexPosition = list(zip(self.images, self.masks))
            random.shuffle(mapIndexPosition)
            self.images, self.masks = zip(*mapIndexPosition)

    def _get_one_hot(self, targets):
        res = np.eye(self.n_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[self.n_classes])

    def _preprocess_image(self, image):

        # Normalize Image
        image =image
        
        #  ImageNet Standardization
        # image = (image - np.min(image)) / (np.max(image)-np.min(image))
        
        # image = (image - np.mean(image)) / np.std(image) 
        return image
    
    def _augmentation(self, image, mask):
        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = self.transform.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        # label = label.astype(np.bool)
        return image, mask
  
    def __data_generation(self, image_batch_list, mask_batch_list):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.target_image_size, self.n_channels))
        y = np.empty((self.batch_size, *self.target_image_size, self.n_classes))
        image  = np.empty((*self.image_size, self.n_channels))

        # Generate data
        for i, image_mask_path in enumerate(zip(image_batch_list, mask_batch_list)):
            img = nib.load(image_mask_path[0]).get_data()
            img = np.clip(img, 20,190)
            image[:,:,0] =img
            image[:,:,1] =img            
            image[:,:,2] =img
            resied_img = image           
            # resied_img = resize(image, [*self.target_image_size, self.n_channels],mode = 'reflect',preserve_range= True)                        
            resied_img = np.asarray(resied_img)

            mask = nib.load(image_mask_path[1]).get_data()
            mask = np.asarray(mask)

            resied_img = self._preprocess_image(resied_img)
            mask_oh = self._get_one_hot(mask)
            # mask = resize(mask, [*self.target_image_size,self.n_classes],mode = 'reflect',preserve_range= True) 
            if mask.any() >= 1:
                if self.transform:
                    resied_img, mask_oh = self._augmentation(resied_img, mask_oh)
            X[i,] = resied_img
            y[i,] = mask_oh
        return X, y




# # Training Data Configuration
# # Data Path
# image_path = '../sliced_img/'
# mask_path = '../sliced_mask/'
# train_data_dir = '../train_data.csv'
# valid_data_dir = '../valid_data.csv'
# augmentation = iaa.SomeOf((0, 3), 
#             [
#                 iaa.Fliplr(0.5),
#                 iaa.Flipud(0.5),
#                 iaa.Noop(),
#                 iaa.OneOf([iaa.Affine(rotate=90),
#                            iaa.Affine(rotate=180),
#                            iaa.Affine(rotate=270)]),
#                 iaa.GaussianBlur(sigma=(0.0, 0.5)),
#             ])
# # Parameters
# train_transform_params = {'image_size': (512,512), 
#                           'target_image_size': (256,256),
#                           'batch_size': 32,
#                           'n_classes': 3,
#                           'n_channels': 3,
#                           'shuffle': True,                          
#                           'transform': augmentation
#                          }

# valid_transform_params = {'image_size': (512,512), 
#                           'target_image_size': (256,256),
#                           'batch_size': 32,
#                           'n_classes': 3,
#                           'n_channels': 3,
#                           'shuffle': True,                         
#                           'transform': None
#                          }
# # Generators
# training_generator = DataGenerator(image_path, mask_path, train_data_dir, **train_transform_params)
# validation_generator = DataGenerator(image_path, mask_path, valid_data_dir, **valid_transform_params)

# # Enable Test Code
# print (training_generator.__len__(), validation_generator.__len__())
# for X, y in training_generator:
#     print (X.shape, y.shape)
# #     imshow(X[0], y[0][:,:,1])

