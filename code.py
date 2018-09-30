import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras import models
from keras import layers


f  = pd.read_csv('complete.csv', encoding='utf-8')
X = f[['age', 'height_cm', 'weight_kg', 'eur_release_clause', 'overall', 'potential', 'pac','sho','pas','dri','def','phy','international_reputation','skill_moves','weak_foot','crossing','finishing','heading_accuracy','short_passing','volleys','dribbling','curve','free_kick_accuracy','long_passing','ball_control','acceleration','sprint_speed','agility','reactions','balance','shot_power','jumping','stamina','strength','long_shots','aggression','interceptions','positioning','vision','penalties','composure','marking','standing_tackle','sliding_tackle','gk_diving','gk_handling','gk_kicking','gk_positioning','gk_reflexes','rs','rw','rf','ram','rcm','rm','rdm','rcb','rb','rwb','st','lw','cf','cam','cm','lm','cdm','cb','lb','lwb','ls','lf','lam','lcm','ldm','lcb','gk']]
Y = f[['eur_value', 'eur_wage']]

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



