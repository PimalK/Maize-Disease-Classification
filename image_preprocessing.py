import numpy as np
import pickle
import cv2
import tensorflow as tf
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
directory_root = 'Combined Data'
def convert_image_to_array(image_dir, default_image_size=(128, 128)):
    """
    Converts an image to a NumPy array after resizing it to the default image size.
    
    :param image_dir: Path to the image file.
    :param default_image_size: Tuple indicating the target image size (width, height).
    :return: NumPy array of the image or None if an error occurs.
    """
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, default_image_size)
            return img_to_array(image)
        return None
    except Exception as e:
        print(f"Error processing image {image_dir}: {e}")
        return None

def load_images_from_directory(directory_root):
    """
    Loads images from a directory structure and returns image arrays and labels.
    
    :param directory_root: Root directory containing class folders with images.
    :return: Tuple (image_list, label_list)
    """
    image_list, label_list = [], []
    try:
        print("[INFO] Loading images ...")
        root_dir = listdir(directory_root)
        root_dir = [d for d in root_dir if d != ".DS_Store"]

        for plant_folder in root_dir:
            plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
            plant_disease_folder_list = [d for d in plant_disease_folder_list if d != ".DS_Store"]

            for plant_disease_folder in plant_disease_folder_list:
                print(f"[INFO] Processing {plant_disease_folder} ...")
                plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
                plant_disease_image_list = [img for img in plant_disease_image_list if img != ".DS_Store"]

                for image in plant_disease_image_list:
                    image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
                    if image_directory.lower().endswith(".jpg"):
                        image_list.append(convert_image_to_array(image_directory))
                        label_list.append(plant_disease_folder)
        print("[INFO] Image loading completed")
    except Exception as e:
        print(f"Error: {e}")
    return image_list, label_list
def preprocess():
    load_images_from_directory(directory_root)
    label_binarizer = LabelBinarizer()
    image_labels = label_binarizer.fit_transform(label_list)
    pickle.dump(label_binarizer,open('label_transform.pkl', 'wb'))
    n_classes = len(label_binarizer.classes_)
    print(label_binarizer.classes_)
    np_image_list = np.array(image_list, dtype=np.float16) / 225.0
    x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.20, random_state = 42)
    aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")
    return x_train, x_test, y_train, y_test, aug