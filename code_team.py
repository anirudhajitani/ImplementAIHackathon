import pandas as pd
import numpy as np
from sklearn import preprocessing

"""
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers
"""

f  = pd.read_csv('top5.csv', encoding='utf-8')
f['win_ratio'] = 0
f['draw_ratio'] = 0
f['loss_ratio'] = 0
f['defence_attr'] = 0
f['attack_attr'] = 0

# For First N teams

for i in range(4) :
	max_ = f['gf'][i*20:(i*20)+20].max()
	min_ = f['gf'][i*20:(i*20)+20].min()
	f['attack_attr'][i*20:(i*20)+20] = ((f['gf'][i*20:(i*20)+20] - min_)/(max_-min_))
	print (f['attack_attr'][i*20:(i*20)+20])
	max_ = f['ga'][i*20:(i*20)+20].max()
	min_ = f['ga'][i*20:(i*20)+20].min()
	f['defence_attr'][i*20:(i*20)+20] = ((max_ - f['ga'][i*20:(i*20)+20])/(max_-min_))
	print (f['defence_attr'][i*20:(i*20)+20])

max_ = f['gf'][80:99].max()
min_ = f['gf'][80:99].min()
f['attack_attr'][80:99] = ((f['gf'][80:99] - min_)/(max_-min_))
print (f['attack_attr'][80:99])
max_ = f['ga'][80:99].max()
min_ = f['ga'][80:99].min()
f['defence_attr'][80:99] = ((max_ - f['ga'][80:99])/(max_-min_))
print (f['defence_attr'][80:99])

#f['win_ratio'] = f['won'] / (f['won'] + f['draw'] + f['loss'])
#f['draw_ratio'] = f['draw'] / (f['won'] + f['draw'] + f['loss'])
#f['loss_ratio'] = f['loss'] / (f['won'] + f['draw'] + f['loss'])

f['win_ratio'][0:80] = f['won'][0:80] / 38
f['draw_ratio'][0:80] = f['draw'][0:80] / 38
f['loss_ratio'][0:80] = f['loss'][0:80] / 38
f['win_ratio'][80:99] = f['won'][80:99] / 34
f['draw_ratio'][80:99] = f['draw'][80:99] / 34
f['loss_ratio'][80:99] = f['loss'][80:99] / 34

"""
for index1, row1 in f.iterrows():
	#f['win_ratio'][index1] = row1['won'] / (row1['won'] + row1['draw'] + row1['loss'])
	#f['draw_ratio'][index1] = row1['draw'] / (row1['won'] + row1['draw'] + row1['loss'])
	#f['loss_ratio'][index1] = row1['loss'] / (row1['won'] + row1['draw'] + row1['loss'])
	f.ix['win_ratio', index1] = float(row1['won']) / (float(row1['won']) + float(row1['draw']) + float(row1['loss']))
	print (f['win_ratio'][index1])
"""

f.to_csv('top5_modified.csv', sep=',', encoding='utf-8')


f1  = pd.read_csv('final_file.csv', encoding='utf-8')
print (f1)




