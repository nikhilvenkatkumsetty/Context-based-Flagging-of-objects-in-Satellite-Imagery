import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

def main():
    model = load_model('trained_bb_classifier.h5')
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #load data
    train_images_path = '/home/carolinemckee/split_train_clean/images/'
    val_images_path = '/home/carolinemckee/split_val_clean_balanced/images/'
    train_labels_path = '/home/carolinemckee/split_train_clean/labels/'
    val_labels_path = '/home/carolinemckee/split_val_clean_balanced/labels/'

    train_images_list = os.listdir(train_images_path) 
    val_images_list = os.listdir(val_images_path)
    print('Number of training images',len(train_images_list))
    print('Number of validation images',len(val_images_list))

    total_train = 0
    for t_img in train_images_list:
        t_labels = train_labels_path + t_img[:-3] + 'txt'
        with open(t_labels,'r') as l_file:
            labels = [line.strip('\n') for line in l_file.readlines()]
            for i in range(0,len(labels),10):
                total_train += 1
    print('Total train labels:',total_train)

    total_val = 0
    for v_img in val_images_list:
        v_labels = val_labels_path + v_img[:-3] + 'txt'
        with open(v_labels,'r') as l_file:
            labels = [line.strip('\n') for line in l_file.readlines()]
            for i in range(0,len(labels),10):
                total_val += 1
    print('Total val labels:',total_val)
    
    train_imgs = np.zeros((total_train,1024,1024,3),dtype='uint8')
    val_imgs = np.zeros((total_val,1024,1024,3),dtype='uint8')
    train_labels = np.zeros((total_train,4),dtype='uint8')
    val_labels = np.zeros((total_val,4),dtype='uint8')
    
    skipped = 0
    img_num = 0
    for t_img in train_images_list:
        img = cv2.imread(train_images_path + t_img,cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t_labels = train_labels_path + t_img[:-3] + 'txt'
        with open(t_labels,'r') as l_file:
            labels = [line.strip('\n') for line in l_file.readlines()]
            for i in range(0,len(labels),10):
                label = labels[i]
                label_list = label.split()
                for j in range(5):
                    if j == 0:
                        label_list[j] = int(label_list[j])
                    else:
                        label_list[j] = int(float(label_list[j])*1024)
                bb_img = np.copy(img)
                if bb_img.shape != (1024,1024,3):
                    skipped += 1
                    continue
                bb_img[int(label_list[2])-int(label_list[4]/2.0):int(label_list[2])+int(label_list[4]/2.0),int(label_list[1]) - int(label_list[3]/2.0):int(label_list[1]) + int(label_list[3]/2.0),:] = np.zeros((1,1,3))
                train_imgs[img_num,:,:,:] = bb_img
                cl = label_list[0]
                one_hot = np.zeros((4,),dtype=int)
                one_hot[cl] = 1
                train_labels[img_num,:] = one_hot
                img_num += 1
        if img_num % 500 == 0:
            print('Added training image',img_num)

    train_imgs = train_imgs[:-skipped]
    train_labels = train_labels[:-skipped]
    train_imgs, train_labels = shuffle(train_imgs, train_labels, random_state=0)
    
    skipped = 0
    img_num = 0
    for v_img in val_images_list:
        img = cv2.imread(val_images_path + v_img,cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        v_labels = val_labels_path + v_img[:-3] + 'txt'
        with open(v_labels,'r') as l_file:
            labels = [line.strip('\n') for line in l_file.readlines()]
            for i in range(0,len(labels),10):
                label = labels[i]
                label_list = label.split()
                for j in range(5):
                    if j == 0:
                        label_list[j] = int(label_list[j])
                    else:
                        label_list[j] = int(float(label_list[j])*1024)
                bb_img = np.copy(img)
                if bb_img.shape != (1024,1024,3):
                    skipped += 1
                    continue
                bb_img[int(label_list[2])-int(label_list[4]/2.0):int(label_list[2])+int(label_list[4]/2.0),int(label_list[1]) - int(label_list[3]/2.0):int(label_list[1]) + int(label_list[3]/2.0),:] = np.zeros((1,1,3))
                val_imgs[img_num,:,:,:] = bb_img
                cl = label_list[0]
                one_hot = np.zeros((4,),dtype=int)
                one_hot[cl] = 1
                val_labels[img_num,:] = one_hot
                img_num += 1
        if img_num % 500 == 0:
            print('Added validation image',img_num)
            
    val_imgs = val_imgs[:-skipped]
    val_labels = val_labels[:-skipped]
    val_imgs, val_labels = shuffle(val_imgs, val_labels, random_state=0)
    
    history = model.fit(train_imgs,train_labels,epochs=3,batch_size=2,validation_data=(val_imgs,val_labels),verbose=True)
    model.save('trained_bb_classifier_retrained.h5')

if __name__ == '__main__':
    main()