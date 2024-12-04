from os import path, mkdir
import tensorflow as tf 
import numpy as np 
import nibabel as nib
from matplotlib import pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import metrics
from models import *
from data_gen_eval import*
import pandas as pd 
from tensorflow.keras import backend as K
import SimpleITK as sitk
from skimage.transform import rescale, resize, downscale_local_mean
from tqdm import tqdm

case_path = '../test_data/'
test_image_path = '../test_sliced_img/'
test_mask_path = '../sliced_mask/'
test_data_dir = '../test_only_data.csv'
models_prediction = '../model_predictions_min_max_epoch_60/'
vol_prediction = '../submit_vol_prediction_min_max_epoch_60/'
if not os.path.exists(vol_prediction):
            os.makedirs(vol_prediction)
test_transform_params = {'image_size': (512,512),
                          'batch_size': 1,
                          'n_classes': 3,
                          'n_channels': 3,
                          'shuffle': False,                         
                          'transform': None
                         }

n = test_transform_params['batch_size']
n_classes = test_transform_params['n_classes']
def get_one_hot(targets):
        res = np.eye(n_classes)[np.array(targets).reshape(-1)]
        return res.reshape(list(targets.shape)+[n_classes])
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # Generators
    # test_generator = DataGenerator(test_image_path, test_mask_path, test_data_dir, **test_transform_params)
    if not os.path.exists(models_prediction):
            os.makedirs(models_prediction)
    model = unet_densenet121((None, None), weights= None)
    model.load_weights('./weights_folder/Densenet_unet_bs_4_epoch_60_min_max_clip_20_190_img_size_512_512_imagenet_fl_kits.h5')
    
    model.summary()

    image = np.empty((512,512,3))

    img_slice_path = pd.read_csv(test_data_dir)
    cases = np.unique([i[0:10] for i in img_slice_path['Image_path']])
    print(len(cases))
    for j in tqdm(cases):
        # print (j)
        case = nib.load(case_path+j+'/imaging.nii.gz')
        case_data = case.get_data()
        case_affine = case.get_affine()
        # # print(case_data.shape[0])
        # label = nib.load(case_path+j+'/segmentation.nii.gz')
        # true_label = label.get_data()
        prediction = np.empty(case_data.shape)
        # # print(prediction.shape)  

        for i in range(case_data.shape[0]):
            img_path =test_image_path+'/'+j+'_slice_'+str(i)+'.nii.gz'         
            # mask_path = test_mask_path+'/'+j+'_slice_'+str(i)+'.nii.gz' 
            im = nib.load(img_path).get_data()            
            img = np.clip(im, 20,190)
            norm_img = (img-np.min(img))/(np.max(img)-np.min(img))
            # img1 = (img-np.mean(img))/(np.std(img))
            image[:,:,0] = norm_img
            image[:,:,1] = norm_img
            image[:,:,2] = norm_img          
            
        ### Enable this line of code for prediction######    
             
            pred = model.predict(image[None,...], batch_size=1, verbose=0, steps=None)
            pred = pred[0]
            pred1 = np.argmax(pred, axis = 2)
            prediction[i,:,:] = pred1
        pred_vol = nib.Nifti1Image(prediction, case_affine)
        pred_vol.set_data_dtype(np.int16)
        nib.save(pred_vol,vol_prediction+'prediction'+j[-6:]+'.nii.gz')


        ##########################################################################
            

        