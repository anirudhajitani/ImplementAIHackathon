import pandas as pd
#from sklearn.cross_validation import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn import linear_model
#from sklearn.model_selection import KFold
#from sklearn.metrics import mean_squared_error
#from sklearn.preprocessing import PolynomialFeatures
#import matplotlib.pyplot as plt
import numpy as np
#from keras.preprocessing.text import Tokenizer
#from keras import models
#from keras import layers

f  = pd.read_csv('complete.csv', encoding='utf-8')
#f  = pd.read_csv('sample1.csv', encoding='utf-8')
#X = f[['age', 'height_cm', 'weight_kg', 'eur_release_clause', 'overall', 'potential', 'pac','sho','pas','dri','def','phy','international_reputation','skill_moves','weak_foot','crossing','finishing','heading_accuracy','short_passing','volleys','dribbling','curve','free_kick_accuracy','long_passing','ball_control','acceleration','sprint_speed','agility','reactions','balance','shot_power','jumping','stamina','strength','long_shots','aggression','interceptions','positioning','vision','penalties','composure','marking','standing_tackle','sliding_tackle','gk_diving','gk_handling','gk_kicking','gk_positioning','gk_reflexes','rs','rw','rf','ram','rcm','rm','rdm','rcb','rb','rwb','st','lw','cf','cam','cm','lm','cdm','cb','lb','lwb','ls','lf','lam','lcm','ldm','lcb','gk']]
#Y = f[['eur_value', 'eur_wage']]

f2 = pd.read_csv('file2.csv', encoding='utf-8')
f2.sort_values(by=['Players'])
f2['height_cm'] = 0
f2['weight_kg'] = 0
f2['eur_release_clause'] = 0
f2['eur_value'] = 0
f2['eur_wage'] = 0
f2['rs'] = 0
f2['rw'] = 0
f2['rf'] = 0
f2['ram'] = 0
f2['rcm'] = 0
f2['rm'] = 0
f2['rdm'] = 0
f2['rcb'] = 0
f2['rb'] = 0
f2['rwb'] = 0
f2['st'] = 0
f2['lw'] = 0
f2['cf'] = 0
f2['cam'] = 0
f2['cm'] = 0
f2['lm'] = 0
f2['cdm'] = 0
f2['cb'] = 0
f2['lb'] = 0
f2['lwb'] = 0
f2['ls'] = 0
f2['lf'] = 0
f2['lam'] = 0
f2['lcm'] = 0
f2['ldm'] = 0
f2['lcb'] = 0
f2['gk'] = 0

f2.to_csv('sample1.csv', sep=',', encoding='utf-8')

for index1, row1 in f.iterrows():
	for index2, row2 in f2.iterrows():
		str1 = str(row2['Players']).split(' ')[-1].encode('utf-8')
		str2 = str(row1['name']).split(' ')[-1].encode('utf-8')
		if str1 == str2 and row1['club'] == row2['club'] :
			print (str(str1), row1['club'])
			print (index1, index2)
			#f2.at['eur_release_clause', index2] = row1['eur_release_clause']
			f2['height_cm'][index2] = row1['height_cm']
			f2['weight_kg'][index2] = row1['weight_kg']
			f2['eur_release_clause'][index2] = row1['eur_release_clause']
			f2['eur_value'][index2] = row1['eur_value']
			f2['eur_wage'][index2] = row1['eur_wage']
			f2['rs'][index2] = row1['rs']
			f2['rw'][index2] = row1['rw']
			f2['rf'][index2] = row1['rf']
			f2['ram'][index2] = row1['ram']
			f2['rcm'][index2] = row1['rcm']
			f2['rm'][index2] = row1['rm']
			f2['rdm'][index2] = row1['rdm']
			f2['rcb'][index2] = row1['rcb']
			f2['rb'][index2] = row1['rb']
			f2['rwb'][index2] = row1['rwb']
			f2['st'][index2] = row1['st']
			f2['lw'][index2] = row1['lw']
			f2['cf'][index2] = row1['cf']
			f2['cam'][index2] = row1['cam']
			f2['cm'][index2] = row1['cm']
			f2['lm'][index2] = row1['lm']
			f2['cdm'][index2] = row1['cdm']
			f2['cb'][index2] = row1['cb']
			f2['lb'][index2] = row1['lb']
			f2['lwb'][index2] = row1['lwb']
			f2['ls'][index2] = row1['ls']
			f2['lf'][index2] = row1['lf']
			f2['lam'][index2] = row1['lam']
			f2['lcm'][index2] = row1['lcm']
			f2['ldm'][index2] = row1['ldm']
			f2['lcb'][index2] = row1['lcb']
			f2['gk'][index2] = row1['gk']

f2.to_csv('final_file.csv', sep=',', encoding='utf-8')

"""
sc = StandardScaler()

X[0:].fillna(0, inplace=True)
#print (X, Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

kfold = KFold(n_splits=10, shuffle=True)
cvloss = []
cvaccuracy = []
for train, test in kfold.split(X, Y):
	# Start neural network
	network = models.Sequential()

# Add fully connected layer with a ReLU activation function
	network.add(layers.Dense(units=1024, activation='relu', input_shape=(X_train.shape[1],)))

# Add fully connected layer with a ReLU activation function
	network.add(layers.Dense(units=1024, activation='relu'))
	network.add(layers.Dense(units=1024, activation='relu'))

# Add fully connected layer with no activation function
	network.add(layers.Dense(units=2))

	network.compile(loss='mean_squared_error', # Mean squared error
                	optimizer='RMSprop', # Optimization algorithm
                	metrics=['accuracy']) # Mean squared error


	history = network.fit(X.iloc[train], # Features
                      	Y.iloc[train], # Target vector
                      	epochs=10, # Number of epochs
                      	verbose=0, # No output
		      	batch_size=100) #Batch size
                      	#validation_data=(X[][test], y[][test])) # Data for evaluation
	loss, accuracy = network.evaluate(X.iloc[test], Y.iloc[test], batch_size=100)
	cvloss.append((loss)**0.5)
	cvaccuracy.append(accuracy)
	print (loss**0.5)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvloss), np.std(cvloss)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvaccuracy), np.std(cvaccuracy)))

al = [0.001, 0.01, 0.1, 1 , 10, 100, 1000, 10000, 100000, 100000]
for a in al :
	reg = linear_model.Ridge(alpha=a, normalize=True)
	reg.fit(X_train, y_train)

	print ("alpha = ", a)
	print ("RMSE Train: ", np.mean((reg.predict(X_train)-y_train)**2) ** 0.5)
	print ("RMSE Test: ", np.mean((reg.predict(X_test)-y_test)**2) ** 0.5)
	#print ("L2 Norm weight Vector: ", np.sum((reg.coef_)**2)** 0.5)
	#print ("Regresssion Intercept ", (reg.intercept_))
	#print ("Regression Weights ", (reg.coef_))
	print ("Regression Score ", reg.score(X_test, y_test))

"""
