import sys, os
from pathlib import Path
path = Path(os.getcwd())
sys.path.append(str(path.parent))


import BayesKeras
from BayesKeras import PosteriorModel
from BayesKeras import analyzers

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import numpy as np


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--opt")
parser.add_argument("--arch")
parser.add_argument("--attack")
parser.add_argument("--G", type=int, nargs='?', const=-1)
parser.add_argument("--R", type=float, nargs='?', const=-1)
parser.add_argument("--N", type=int, nargs='?', const=-1)
args = parser.parse_args()
opt = str(args.opt)
arch = str(args.arch)
attack = str(args.attack)

inference = opt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train/255.
X_test = X_test/255.
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

model_type = arch

model = PosteriorModel("%s_%s_Posterior_0"%(inference, model_type))
from tqdm import trange

loss = tf.keras.losses.SparseCategoricalCrossentropy()

if(attack == "FGSM"):
    meth = analyzers.FGSM
elif(attack == "PGD"):
    meth = analyzers.PGD
elif(attack == "CW"):
    meth = analyzers.CW
elif(attack == 'GA'):
    meth = analyzers.cifar_gen_attack

num_images = 100

#print(model._predict(X_test[0:1]))
#print(analyzers.IBP(model, X_test[0:1], eps=0.0, weights=model.model.get_weights(), predict=True))
accuracy = tf.keras.metrics.Accuracy()
preds = model.predict( X_test[0:num_images])
accuracy.update_state(np.argmax(preds, axis=1), y_test[0:num_images])
print("Accuracy: ",  accuracy.result())
accuracy = tf.keras.metrics.Accuracy()
if attack == 'GA':
    adv = meth(model, X_test[0:num_images], G=args.G, R=args.R, N=args.N, D=0.007)
else:
    adv = meth(model, X_test[0:num_images], eps=0.007, loss_fn=loss, num_models=5)
preds = model.predict(adv) #np.argmax(model.predict(np.asarray(adv)), axis=1)
accuracy.update_state(np.argmax(preds, axis=1), y_test[0:num_images])
fgsm = accuracy.result()
print(attack + " Robustness: ", accuracy.result())

accuracy = tf.keras.metrics.Accuracy()
preds = analyzers.chernoff_bound_verification(model, X_test[0:num_images], 0.0035, y_test[0:num_images], confidence=0.90)
#print(preds.shape)
#print(np.argmax(preds, axis=1).shape)
accuracy.update_state(np.argmax(preds, axis=1), y_test[0:num_images])
print("Massart Lower Bound (IBP): ",  accuracy.result())


