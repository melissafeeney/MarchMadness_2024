#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:14:08 2024

@author: mfeene
"""

# -------------------------
# III. GENERATE MODEL WITHOUT ATTENTION MECHANISM
# -------------------------

# Run first for reproducability
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(1)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout, Dense, Concatenate, Attention, Reshape, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from matplotlib import pyplot
import tensorflow.keras.models  
from tensorflow.keras.utils import plot_model


# -------------------------
# Definitions
# -------------------------
threshold = 0.5
precision = tf.keras.metrics.Precision(thresholds = threshold)
recall = tf.keras.metrics.Recall(thresholds = threshold)
roc_auc = tf.keras.metrics.AUC(curve = 'ROC') # roc auc
f1_macro = tf.keras.metrics.F1Score(average = 'macro', threshold = threshold)
#earlyStopping = EarlyStopping(monitor = 'val_accuracy', mode = 'max', verbose = 1, patience = 5) # early stopping to optimize training
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 3, min_lr = 0.000001)
hidden_layer_act = tf.keras.layers.LeakyReLU(alpha = 0.01)



# -------------------------
# Data Processing
# -------------------------
final_model_data = pd.read_csv('/content/final_model_data_addl.csv')

# Split into X and Y
X = final_model_data.iloc[:, 4:-1].values
y = final_model_data.iloc[:, -1].values

# Create a train-test split with ratio 30:70
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2, random_state = 123)

# Scale the training and testing data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Save the scaler for future use
scalerfile = 'scaler.save'
pickle.dump(sc, open(scalerfile, 'wb'))

# for the f1 score to work
y_train = tf.cast(y_train, tf.float32)
y_test = tf.cast(y_test, tf.float32)



# -------------------------
# Designing the submodels
# -------------------------
def fit_model(X_train, y_train):
  # define model
  model = Sequential()
  model.add(Dense(32, input_dim=X_train.shape[1], activation = hidden_layer_act))  
  model.add(Dense(16, activation = hidden_layer_act))
  model.add(BatchNormalization())
  model.add(Dense(8, activation = hidden_layer_act))
  model.add(Dense(1, activation = 'sigmoid'))
  model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1_macro, recall, precision, roc_auc])
  # fit model
  model.fit(X_train, y_train, validation_split = 0.2, epochs = 10, verbose = 2, callbacks = [reduce_lr])
  return model

##### create models/ directory

# submodels
n_members = 5
for i in range(n_members):
	# fit model
	model = fit_model(X_train, y_train)
	# save model
	filename = 'models/model_' + str(i + 1) + '.keras' # save weights
	model.save(filename)
	print('>Saved %s' % filename)
    
    
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		# define filename for this ensemble
		filename = 'models/model_' + str(i + 1) + '.keras'
		# load model from file
		model = tensorflow.keras.models.load_model(filename)
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models

# load all models
n_members = 5
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

# evaluate standalone models on test dataset
for model in members:
  print(model.evaluate(X_test, y_test, verbose = 2))
  
  
  
# -------------------------
# Model blending
# -------------------------  
# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		# make prediction
		yhat = model.predict(inputX, verbose=0)
		# stack predictions into [rows, members, probabilities]
		if stackX is None:
			stackX = yhat
		else:
			stackX = np.dstack((stackX, yhat))
	# flatten predictions to [rows, members x probabilities]
	stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
	return stackX


# Stacking
# update all layers in all models to not be trainable
for i in range(len(members)):
	model = members[i]
	for layer in model.layers:
		# make not trainable
		layer.trainable = False
		# rename to avoid 'unique layer name' issue
		layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
        
        
def define_stacked_model(members):
  # update all layers in all models to not be trainable
  for i in range(len(members)):
    model = members[i]
    for layer in model.layers:
      # make not trainable
      layer.trainable = False
      # rename to avoid 'unique layer name' issue
      layer._name = 'ensemble_' + str(i+1) + '_' + layer.name

  # define multi-headed input
  ensemble_visible = [model.input for model in members]

  # concatenate outputs within the model using Concatenate layer
  merged = Concatenate(axis=1)(ensemble_visible)

  # add new layers
  hidden = Dense(64, activation = hidden_layer_act)(merged)
  hidden = Dense(32, activation = hidden_layer_act)(hidden)
  hidden = Dense(16, activation = hidden_layer_act)(hidden)
  hidden = Dense(8, activation = hidden_layer_act)(hidden)
  output = Dense(1, activation='sigmoid')(hidden)
  model = Model(inputs=ensemble_visible, outputs=output)

  # plot graph of ensemble
  #plot_model(model, show_shapes=True, to_file='model_graph.png')
  # compile
  model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1_macro, recall, precision, roc_auc])
  return model

# define ensemble model
stacked_model = define_stacked_model(members)


plot_model(stacked_model, show_shapes = True, to_file = 'model_graph.png')


# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
	X = [inputX for _ in range(len(model.input))]
	model.fit(X, inputy, epochs = 10, verbose = 2)
    
    
# fit stacked model on test dataset
fit_stacked_model(stacked_model, X_test, y_test)


# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
	# prepare input data
	X = [inputX for _ in range(len(model.input))]
	# make prediction
	return model.predict(X, verbose = 2)



# -------------------------
# Evaluating the Model on Test data
# -------------------------
y_preds = predict_stacked_model(stacked_model, X_test)
y_preds_clean = np.where(y_preds >= 0.5, 1, 0)

print('Non-Attention Model: ')
print(' ')
# Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy: ', accuracy_score(y_test, y_preds_clean))

# F1 Score
from sklearn.metrics import f1_score
print('F1 Score: ', f1_score(y_test, y_preds_clean, average = 'macro'))

# Recall
from sklearn.metrics import recall_score
print('Recall: ', recall_score(y_test, y_preds_clean))

# Precision
from sklearn.metrics import precision_score
print('Precision: ', precision_score(y_test, y_preds_clean))

# ROC AUC
from sklearn.metrics import roc_auc_score
print('ROC AUC: ', roc_auc_score(y_test, y_preds_clean))

print('') 

# Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_preds_clean))

# Save model as pkl file
with open('stacked_model.pkl', 'wb') as fid:
    pickle.dump(stacked_model, fid)
    
    
