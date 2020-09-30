# Author: Matthew Wicker

# Description: Minimal working example of training and saving
# a BNN trained with Bayes by backprop (BBB)
# can handle any Keras model
import sys, os
from pathlib import Path
path = Path(os.getcwd())
sys.path.append(str(path.parent))

import BayesKeras
import BayesKeras.optimizers as optimizers

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


#tf.debugging.set_log_device_placement(True)
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--eps")
parser.add_argument("--lam")
parser.add_argument("--depth")
parser.add_argument("--opt")

args = parser.parse_args()
eps = float(args.eps)
lam = float(args.lam)
optim = str(args.opt)
depth = int(args.depth)
rob = 0

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train/255.
X_test = X_test/255.
X_train = X_train.astype("float32").reshape(-1, 28*28)
X_test = X_test.astype("float32").reshape(-1, 28* 28)

model = Sequential()
model.add(Dense(256, input_shape=(28*28,), activation="relu", dtype='float32'))
for i in range(depth-1):
    model.add(Dense(256, activation="relu"))
model.add(Dense(10, activation="softmax"))

inf = 20
full_covar = False
if(optim == 'VOGN'):
    learning_rate = 0.25; decay=0.0
    opt = optimizers.VariationalOnlineGuassNewton()
elif(optim == 'BBB'):
    inf = 2; learning_rate = 0.25
    #inf = 2.5; learning_rate = 0.5
    decay=0.0
    opt = optimizers.BayesByBackprop()
elif(optim == 'SWAG'):
    learning_rate = 0.01; decay=0.0
    opt = optimizers.StochasticWeightAveragingGaussian()
elif(optim == 'SWAG-FC'):
    learning_rate = 0.01; decay=0.0; full_covar=True
    opt = optimizers.StochasticWeightAveragingGaussian()
elif(optim == 'SGD'):
    learning_rate = 0.035; decay=0.0
    opt = optimizers.StochasticGradientDescent()
elif(optim == 'NA'):
    learning_rate = 0.001; decay=0.0
    opt = optimizers.NoisyAdam()

# Compile the model to train with Bayesian inference
if(rob == 0):
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
elif(rob != 0):
    loss = BayesKeras.optimizers.losses.robust_crossentropy_loss

bayes_model = opt.compile(model, loss_fn=loss, epochs=15, learning_rate=learning_rate, full_covar=full_covar,
                          decay=decay, robust_train=rob, inflate_prior=inf)

# Train the model on your data
bayes_model.train(X_train, y_train, X_test, y_test)

# Save your approxiate Bayesian posterior
bayes_model.save("%s_FCN_Posterior_%s"%(optim, depth))

