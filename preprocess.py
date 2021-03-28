import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import sys

from PIL import Image
from numpy import asarray 
from PIL import ImageOps
from texttable import Texttable

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score


image_path = r'.\static'
models_path = r'.\models'

def iterate_through_images():
    images = os.listdir(image_path)
    image_df = pd.DataFrame(images)
    image_df.columns=["image_name"]
    return image_df

	
def load_image(image_path,df,index,verbose = True):
    path = os.path.join(image_path,df['image_name'].iloc[index])
    img=Image.open(path)
    if verbose:
        print(path)
        print(img.format)
        print(img.size)
        print(img.mode)
    return img

	
def preprocess_image(images_directory = image_path,annotation='False',size = 16,equalize = True,as_df=True):
    images_data = []
    image_as_rows = []
    df = iterate_through_images()
    for i in range(df.shape[0]):
        image = load_image(images_directory,df,i,verbose= False)

        # if annotation:
        #     data = list(df[['x1','y1','x2','y2']].iloc[i])
        #     image = image.crop(box=(data[0],data[1],data[2],data[3]))

        image = image.resize((size,size))
        image = ImageOps.equalize(image)
        image_data =  asarray(image,dtype='int32')
        row =  image_data.ravel().tolist()
        images_data.append(image_data)
        image_as_rows.append(row)
    images_df = pd.DataFrame.from_records(image_as_rows)
    images_data = np.array(images_data)

    if as_df:
        return images_data,images_df
    else:
        return images_data
