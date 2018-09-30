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
X = f[['age', 'height_cm', 'weight_kg', 'eur_release_clause', 'overall', 'pac','sho','pas','dri','def','phy','international_reputation','skill_moves','weak_foot','crossing','finishing','heading_accuracy','short_passing','volleys','dribbling','curve','free_kick_accuracy','long_passing','ball_control','acceleration','sprint_speed','agility','reactions','balance','shot_power','jumping','stamina','strength','long_shots','aggression','interceptions','positioning','vision','penalties','composure','marking','standing_tackle','sliding_tackle','gk_diving','gk_handling','gk_kicking','gk_positioning','gk_reflexes','rs','rw','rf','ram','rcm','rm','rdm','rcb','rb','rwb','st','lw','cf','cam','cm','lm','cdm','cb','lb','lwb','ls','lf','lam','lcm','ldm','lcb','gk']]
Y = f[['eur_value', 'eur_wage']]

#print ("WORKING")
#print (type(X), type(Y))

sc = StandardScaler()

X[0:].fillna(0, inplace=True)
#print (X, Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#"""
kfold = KFold(n_splits=10, shuffle=True)
cvloss = []
cvaccuracy = []
for train, test in kfold.split(X, Y):
	# Start neural network
	network = models.Sequential()

# Add fully connected layer with a ReLU activation function
	network.add(layers.Dense(units=1024, activation='relu', input_shape=(X.iloc[test].shape[1],)))

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
	print ("RMSE", loss**0.5)
	print ("Acc :", accuracy)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvloss), np.std(cvloss)))
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvaccuracy), np.std(cvaccuracy)))
#"""
al = [0.001, 0.01, 0.1, 1 , 10, 100, 1000, 10000, 100000, 100000]
for a in al :
	reg = linear_model.Ridge(alpha=a, normalize=True)
	reg.fit(X_train, y_train)

	print ("alpha = ", a)
	print ("WORK ", type(reg.predict(X_train)), type(y_train))
	print ("RMSE Train: ", np.mean((reg.predict(X_train)-y_train)**2) ** 0.5)
	print ("RMSE Test: ", np.mean((reg.predict(X_test)-y_test)**2) ** 0.5)
	#print ("L2 Norm weight Vector: ", np.sum((reg.coef_)**2)** 0.5)
	#print ("Regresssion Intercept ", (reg.intercept_))
	#print ("Regression Weights ", (reg.coef_))
	print ("Regression Score ", reg.score(X_test, y_test))

reg = linear_model.Ridge(alpha=1, normalize=True)
reg.fit(X_train, y_train)
#"""
#"""
df  = pd.read_csv('final.csv', encoding='utf-8')
df[0:].fillna(0, inplace=True)

#for index, row in df.iterrows():
#	if row['eur_value'] == 0 :
#


df = df[df.eur_value != 0]
df = df[df.eur_release_clause != 0]

df['impact_score'] = 0
#print (df)
#"""
X_ = df[['age', 'height_cm', 'weight_kg', 'eur_release_clause', 'overall', 'pac','sho','pas','dri','def','phy','international_reputation','skill_moves','weak_foot','crossing','finishing','heading_accuracy','short_passing','volleys','dribbling','curve','free_kick_accuracy','long_passing','ball_control','acceleration','sprint_speed','agility','reactions','balance','shot_power','jumping','stamina','strength','long_shots','aggression','interceptions','positioning','vision','penalties','composure','marking','standing_tackle','sliding_tackle','gk_diving','gk_handling','gk_kicking','gk_positioning','gk_reflexes','rs','rw','rf','ram','rcm','rm','rdm','rcb','rb','rwb','st','lw','cf','cam','cm','lm','cdm','cb','lb','lwb','ls','lf','lam','lcm','ldm','lcb','gk']]
Y_ = df[['eur_value', 'eur_wage']]

print (X_)
print ("RMSE Test: ", np.mean((reg.predict(X_)-Y_)**2) ** 0.5)
print ("Regression Score ", reg.score(X_, Y_))
loss, accuracy = network.evaluate(X_, Y_)
print ("Loss : ")
print (loss**0.5)
print ("Accuracy ")
print (accuracy)
#"""

X__1 = df.iloc[:,55:73]
#print (X__1)
Y__1 = df.iloc[:,73]
#y__1 = df.iloc[['16/17.18']]
print (Y__1)
X__2 = df.iloc[:,74:92]
#print (X__2)
Y__2 = df.iloc[:,92]
#y__2 = df.iloc[['15/16.18']]
#print (Y__2)
X__3 = df.iloc[:,93:111]
#print (X__3)
Y__3 = df.iloc[:,111]
#y__3 = df.iloc[['14/15.18']]
#print (Y__3)
X__4 = df.iloc[:,112:130]
#print (X__4)
Y__4 = df.iloc[:,130]
#y__4 = df.iloc[['13/14.18']]
#print (Y__4)
X__5 = df.iloc[:,131:149]
#print (X__5)
Y__5 = df.iloc[:,149]
#y__5 = df.iloc[['13/14.18']]
#print (Y__5)


#df['impact_score'] = 5*df.iloc[:,55] + 4*df_iloc[:,74] + 3*df_iloc[:,93] + 2*df_iloc[:,111] + 1*df_iloc[:,131]
df['impact_score'] = 5*Y__1 + 4*Y__2 + 3*Y__3 + 2*Y__4 + 1*Y__5
#df['impact_score'] = 5*Y__1 + 4*Y__2 + 3*Y__3 + 2*Y__4
#print ("IMPACT SCORE")
#print (df['impact_score'])

y__1 = Y__1.to_frame()
y__2 = Y__2.to_frame()
y__3 = Y__3.to_frame()
y__4 = Y__4.to_frame()
y__5 = Y__5.to_frame()

X1_train, X1_test, y1_train, y1_test = train_test_split(X__1, y__1, test_size=0.20)
X2_train, X2_test, y2_train, y2_test = train_test_split(X__2, y__2, test_size=0.20)
X3_train, X3_test, y3_train, y3_test = train_test_split(X__3, y__3, test_size=0.20)
X4_train, X4_test, y4_train, y4_test = train_test_split(X__4, y__4, test_size=0.20)
X5_train, X5_test, y5_train, y5_test = train_test_split(X__5, y__5, test_size=0.20)

#NEURAL NETWORKS
#"""
kfold = KFold(n_splits=5, shuffle=True)
cvloss = []
cvaccuracy = []
for train, test in kfold.split(X__1, y__1):
	# Start neural network
	network = models.Sequential()

# Add fully connected layer with a ReLU activation function
	network.add(layers.Dense(units=1024, activation='relu', input_shape=(X__1.shape[1],)))

# Add fully connected layer with a ReLU activation function
	network.add(layers.Dense(units=1024, activation='relu'))
	network.add(layers.Dense(units=1024, activation='relu'))
	network.add(layers.Dense(units=1024, activation='relu'))
	network.add(layers.Dense(units=1024, activation='relu'))

# Add fully connected layer with no activation function
	network.add(layers.Dense(units=1))

	network.compile(loss='mean_squared_error', # Mean squared error
                	optimizer='RMSprop', # Optimization algorithm
                	metrics=['accuracy']) # Mean squared error


	history = network.fit(X__1.iloc[train], # Features
                      	Y__1.iloc[train], # Target vector
                      	epochs=10, # Number of epochs
                      	verbose=0, # No output
		      	batch_size=100) #Batch size
                      	#validation_data=(X[][test], y[][test])) # Data for evaluation
	loss, accuracy = network.evaluate(X__1.iloc[test], Y__1.iloc[test], batch_size=100)
	cvloss.append((loss)**0.5)
	cvaccuracy.append(accuracy)
	print ("Loss", loss**0.5)
	print ("Accuracy", accuracy)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvloss), np.std(cvloss)))


kfold = KFold(n_splits=5, shuffle=True)
cvloss = []
cvaccuracy = []
for train, test in kfold.split(X__2, y__2):
	# Start neural network
	network = models.Sequential()

# Add fully connected layer with a ReLU activation function
	network.add(layers.Dense(units=1024, activation='relu', input_shape=(X__2.shape[1],)))

# Add fully connected layer with a ReLU activation function
	network.add(layers.Dense(units=1024, activation='relu'))
	network.add(layers.Dense(units=1024, activation='relu'))

# Add fully connected layer with no activation function
	network.add(layers.Dense(units=1))

	network.compile(loss='mean_squared_error', # Mean squared error
                	optimizer='RMSprop', # Optimization algorithm
                	metrics=['accuracy']) # Mean squared error


	history = network.fit(X__2.iloc[train], # Features
                      	Y__2.iloc[train], # Target vector
                      	epochs=10, # Number of epochs
                      	verbose=0, # No output
		      	batch_size=100) #Batch size
                      	#validation_data=(X[][test], y[][test])) # Data for evaluation
	loss, accuracy = network.evaluate(X__2.iloc[test], Y__2.iloc[test], batch_size=100)
	cvloss.append((loss)**0.5)
	cvaccuracy.append(accuracy)
	print ("Loss", loss**0.5)
	print ("Accuracy", accuracy)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvloss), np.std(cvloss)))


kfold = KFold(n_splits=5, shuffle=True)
cvloss = []
cvaccuracy = []
for train, test in kfold.split(X__3, y__3):
	# Start neural network
	network = models.Sequential()

# Add fully connected layer with a ReLU activation function
	network.add(layers.Dense(units=1024, activation='relu', input_shape=(X__3.shape[1],)))

# Add fully connected layer with a ReLU activation function
	network.add(layers.Dense(units=1024, activation='relu'))
	network.add(layers.Dense(units=1024, activation='relu'))
	network.add(layers.Dense(units=1024, activation='relu'))
	network.add(layers.Dense(units=1024, activation='relu'))

# Add fully connected layer with no activation function
	network.add(layers.Dense(units=1))

	network.compile(loss='mean_squared_error', # Mean squared error
                	optimizer='RMSprop', # Optimization algorithm
                	metrics=['accuracy']) # Mean squared error


	history = network.fit(X__3.iloc[train], # Features
                      	Y__3.iloc[train], # Target vector
                      	epochs=10, # Number of epochs
                      	verbose=0, # No output
		      	batch_size=100) #Batch size
                      	#validation_data=(X[][test], y[][test])) # Data for evaluation
	loss, accuracy = network.evaluate(X__3.iloc[test], Y__3.iloc[test], batch_size=100)
	cvloss.append((loss)**0.5)
	cvaccuracy.append(accuracy)
	print ("Loss", loss**0.5)
	print ("Accuracy", accuracy)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvloss), np.std(cvloss)))


kfold = KFold(n_splits=5, shuffle=True)
cvloss = []
cvaccuracy = []
for train, test in kfold.split(X__4, y__4):
	# Start neural network
	network = models.Sequential()

# Add fully connected layer with a ReLU activation function
	network.add(layers.Dense(units=1024, activation='relu', input_shape=(X__4.shape[1],)))

# Add fully connected layer with a ReLU activation function
	network.add(layers.Dense(units=1024, activation='relu'))
	network.add(layers.Dense(units=1024, activation='relu'))
	network.add(layers.Dense(units=1024, activation='relu'))
	network.add(layers.Dense(units=1024, activation='relu'))

# Add fully connected layer with no activation function
	network.add(layers.Dense(units=1))

	network.compile(loss='mean_squared_error', # Mean squared error
                	optimizer='RMSprop', # Optimization algorithm
                	metrics=['accuracy']) # Mean squared error


	history = network.fit(X__4.iloc[train], # Features
                      	Y__4.iloc[train], # Target vector
                      	epochs=10, # Number of epochs
                      	verbose=0, # No output
		      	batch_size=100) #Batch size
                      	#validation_data=(X[][test], y[][test])) # Data for evaluation
	loss, accuracy = network.evaluate(X__4.iloc[test], Y__4.iloc[test], batch_size=100)
	cvloss.append((loss)**0.5)
	cvaccuracy.append(accuracy)
	print ("Loss", loss**0.5)
	print ("Accuracy", accuracy)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvloss), np.std(cvloss)))

kfold = KFold(n_splits=5, shuffle=True)
cvloss = []
cvaccuracy = []
for train, test in kfold.split(X__5, y__5):
	# Start neural network
	network = models.Sequential()

# Add fully connected layer with a ReLU activation function
	network.add(layers.Dense(units=1024, activation='relu', input_shape=(X__5.shape[1],)))

# Add fully connected layer with a ReLU activation function
	network.add(layers.Dense(units=1024, activation='relu'))
	network.add(layers.Dense(units=1024, activation='relu'))
	network.add(layers.Dense(units=1024, activation='relu'))
	network.add(layers.Dense(units=1024, activation='relu'))

# Add fully connected layer with no activation function
	network.add(layers.Dense(units=1))

	network.compile(loss='mean_squared_error', # Mean squared error
                	optimizer='RMSprop', # Optimization algorithm
                	metrics=['accuracy']) # Mean squared error


	history = network.fit(X__5.iloc[train], # Features
                      	Y__5.iloc[train], # Target vector
                      	epochs=10, # Number of epochs
                      	verbose=0, # No output
		      	batch_size=100) #Batch size
                      	#validation_data=(X[][test], y[][test])) # Data for evaluation
	loss, accuracy = network.evaluate(X__5.iloc[test], Y__5.iloc[test], batch_size=100)
	cvloss.append((loss)**0.5)
	cvaccuracy.append(accuracy)
	print ("Loss", loss**0.5)
	print ("Accuracy", accuracy)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvloss), np.std(cvloss)))
#"""

#Regression goes to infinity sometimes
#"""
#print (type(X1_train), type(X1_test), type(y1_train), type(y1_test))
reg = linear_model.Ridge(alpha=1, normalize=True)
reg.fit(X1_train, y1_train)
pred = reg.predict(X1_train)
print ("RMSE Test 2016-17: ", np.mean((reg.predict(X1_train)-y1_train.values)**2) ** 0.5)
print ("RMSE Test: ", np.mean((reg.predict(X1_test)-y1_test.values)**2) ** 0.5)
print ("Regression Score ", reg.score(X1_test, y1_test))

reg = linear_model.Ridge(alpha=1, normalize=True)
reg.fit(X2_train, y2_train)
pred = reg.predict(X2_train)
print ("RMSE Test 2015-16: ", np.mean((reg.predict(X2_train)-y2_train.values)**2) ** 0.5)
print ("RMSE Test: ", np.mean((reg.predict(X2_test)-y2_test.values)**2) ** 0.5)
print ("Regression Score ", reg.score(X2_test, y2_test))

reg = linear_model.Ridge(alpha=1, normalize=True)
reg.fit(X1_train, y1_train)
pred = reg.predict(X1_train)
print ("RMSE Test 2014-15: ", np.mean((reg.predict(X3_train)-y3_train.values)**2) ** 0.5)
print ("RMSE Test: ", np.mean((reg.predict(X3_test)-y3_test.values)**2) ** 0.5)
print ("Regression Score ", reg.score(X3_test, y3_test))

reg = linear_model.Ridge(alpha=1, normalize=True)
reg.fit(X4_train, y4_train)
pred = reg.predict(X4_train)
print ("RMSE Test 2013-14: ", np.mean((reg.predict(X4_train)-y4_train.values)**2) ** 0.5)
print ("RMSE Test: ", np.mean((reg.predict(X4_test)-y4_test.values)**2) ** 0.5)
print ("Regression Score ", reg.score(X4_test, y4_test))

reg = linear_model.Ridge(alpha=1, normalize=True)
reg.fit(X5_train, y5_train)
pred = reg.predict(X5_train)
print ("RMSE Test 2012-13: ", np.mean((reg.predict(X5_train)-y5_train.values)**2) ** 0.5)
print ("RMSE Test: ", np.mean((reg.predict(X5_test)-y5_test.values)**2) ** 0.5)
print ("Regression Score ", reg.score(X5_test, y5_test))
#"""

#X_.drop(columns =['Unnamed:0','Players','club','league','nationality', 'Position'])
#X_[0:].fillna(0, inplace=True)

teams  = pd.read_csv('top5_modified.csv', encoding='utf-8')
teams[0:].fillna(0, inplace=True)
print (teams)
teams['Avg_Impact_Score_Gk'] = 0
teams['Avg_Impact_Score_Def'] = 0
teams['Avg_Impact_Score_Mid'] = 0
teams['Avg_Impact_Score_Fwd'] = 0
avg_df = 0
count_df = 0
avg_gk = 0
count_gk = 0
avg_mf = 0
count_mf = 0
avg_fw = 0
count_fw = 0

max_ = df['impact_score'].max()
min_ = df['impact_score'].min()
#print (max_, min_)
df['impact_score_norm'] = ((df['impact_score'] - min_)/(max_-min_))
print(df['impact_norm'])
print(df['impact_score_norm'])

max_ = df['overall'].max()
min_ = df['overall'].min()
#print (max_, min_)
df['overall_norm'] = ((df['overall'] - min_)/(max_-min_))

#print ("Overall Norm")
#print (df['overall_norm'])
#print ("Impact_Score Norm")
#print (df['impact_score_norm'])

with open('impact_score.txt', mode='a') as f :
	for inx, r in teams.iterrows():
		avg_gk = 0.0
		count_gk = 0
		avg_df = 0.0
		count_df = 0
		avg_mf = 0.0
		count_mf = 0
		avg_fw = 0.0
		count_fw = 0
		for inx1, r1 in df.iterrows():
		#print (r['Team'], r1['club'])
			if r['Team'] == r1['club'] and r1['Position'] == 'Goalkeeper' :
				avg_gk = avg_gk + (float(r1['overall_norm']) + float(r1['impact_score_norm']))/2
				count_gk = count_gk + 1
			if r['Team'] == r1['club'] and r1['Position'] == 'Defender' :
				avg_df = avg_df + (float(r1['overall_norm']) + float(r1['impact_score_norm']))/2
				count_df = count_df + 1
			if r['Team'] == r1['club'] and r1['Position'] == 'Midfielder' :
				avg_mf = avg_mf + (float(r1['overall_norm']) + float(r1['impact_score_norm']))/2
				count_mf = count_mf + 1
			if r['Team'] == r1['club'] and r1['Position'] == 'Forward' :
				avg_fw = avg_fw + (float(r1['overall_norm']) + float(r1['impact_score_norm']))/2
				count_fw = count_fw + 1
			print ("PLAYER: ", r1['Player'][inx1])
			print ("IMPACT SCORE: ", r1['impact_score_norm'][inx1])
		if count_gk == 0 or count_df == 0 or count_mf == 0 or count_fw == 0 :
			teams['Avg_Impact_Score_Gk'][inx] = 0 
			teams['Avg_Impact_Score_Def'][inx] = 0
			teams['Avg_Impact_Score_Mid'][inx] = 0
			teams['Avg_Impact_Score_Fwd'][inx] = 0
		else :
			agk = avg_gk/float(count_gk)
			teams['Avg_Impact_Score_Gk'][inx] = agk 
			adf = avg_df/float(count_df)
			teams['Avg_Impact_Score_Def'][inx] = adf
			amf = avg_mf/float(count_mf)
			teams['Avg_Impact_Score_Mid'][inx] = amf
			afw = avg_fw/float(count_fw)
			teams['Avg_Impact_Score_Fwd'][inx] = afw
			print ("Team : ", teams['Team'][inx])
			print ("GK : ", agk, teams['Avg_Impact_Score_Gk'][inx])
			print ("DEF : ", adf, teams['Avg_Impact_Score_Def'][inx])
			print ("MID : ", amf, teams['Avg_Impact_Score_Mid'][inx])
			print ("FWD : ", afw, teams['Avg_Impact_Score_Fwd'][inx])
			f.write(str(agk))
			f.write(",")
			f.write(str(adf))
			f.write(",")
			f.write(str(amf))
			f.write(",")
			f.write(str(afw))
			f.write("\n")
	
#teams = teams[teams.Avg_Impact_Score_Gk != 0]
#teams = teams[teams.Avg_Impact_Score_Def != 0]
#teams = teams[teams.Avg_Impact_Score_Mid != 0]
#teams = teams[teams.Avg_Impact_Score_Fwd != 0]

#print (teams)
#print (teams['Avg_Impact_Score_Gk'])
#print (teams['Avg_Impact_Score_Def'])
#print (teams['Avg_Impact_Score_Mid'])
#print (teams['Avg_Impact_Score_Fwd'])


