import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import sys
import os
import glob

from PIL import Image
from numpy import asarray 
from PIL import ImageOps
from texttable import Texttable
import pickle

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import f1_score, accuracy_score, classification_report, precision_score, recall_score


from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from preprocess import *

app = Flask(__name__)


image_path = r'.\static'
models_path = r'.\models'

## LOAD the Models 
classic_model = pickle.load(open('models\svm_model.pkl', 'rb'))
cnn_model = load_model('models\CNNModel.h5')
encoder = pickle.load(open('models\label_encoder_for_cnn.pkl', 'rb'))
#cnn_model._make_predict_function()



def display_image(image_path):
    path = os.path.join(image_path, os.listdir(image_path)[0])
    img=Image.open(path)
    return img 


# def predict_label(image_info):
# 	i = image.load_img(img_path, target_size=(100,100))
# 	i = image.img_to_array(i)
# 	i = i.reshape(1, 100,100,3)
# 	p = model.predict_classes(i)
# 	return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def homepage():
	return render_template("home.html")

@app.route("/about")
def about_page():
	return " Detectors Trained and service developed by Yuvaraj"


@app.route("/submit", methods = ['GET', 'POST'])
def get_results():

	if request.method == 'POST':
		#files = glob.glob(image_path)
		for f in os.listdir(image_path):
			os.remove(os.path.join(image_path,f))
		img = request.files['my_image']
		option = request.form['options']
		img_path = os.path.join(image_path,img.filename)
		img.save(img_path)
		
		images_data,images_df = preprocess_image()
		

		if option == 'svm':
			prediction = classic_model.predict(images_df)[0]
		if option == 'cnn':
			predicted_label = cnn_model.predict_classes(images_data)
			prediction = encoder.inverse_transform(predicted_label)[0]


	print(str(os.path.join(image_path, os.listdir(image_path)[0])))
	return render_template("home.html", prediction = prediction, img_path = img_path )





if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)