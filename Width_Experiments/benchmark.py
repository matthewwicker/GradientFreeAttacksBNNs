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

import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--attack")
parser.add_argument("--opt")
parser.add_argument("--G", type=int)
parser.add_argument("--R", type=float)
parser.add_argument("--N", type=int)

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2'

widths = ["128", "256", "512", "1024"]

args = parser.parse_args()
attack = str(args.attack)
opt = str(args.opt)
inference = opt
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train/255.
X_test = X_test/255.
X_train = X_train.astype("float32").reshape(-1, 28*28)
X_test = X_test.astype("float32").reshape(-1, 28* 28)

num_images = 100
ord = 1
from tqdm import trange

if(attack == "FGSM"):
    meth = analyzers.FGSM
elif(attack == "PGD"):
    meth = analyzers.PGD
elif(attack == "CW"):
    meth = analyzers.CW
elif(attack == 'GA'):
    meth = analyzers.gen_attack

results = []; acc = []; veri = []
for w in widths:
    if(inference == 'SGD'):
        det = 1
    else:
        det = 0 
    try:
        model = PosteriorModel("%s_FCN_Posterior_%s"%(opt, w), deterministic=det)
    except:
        results.append(float('NaN'))
        continue
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    if(inference == "SWAG-FC"):
        results.append(float('NaN'))
        continue

    """
    accuracy = tf.keras.metrics.Accuracy()
    preds = model.predict(X_test[0:num_images]) #np.argmax(model.predict(np.asarray(adv)), axis=1)
    accuracy.update_state(np.argmax(preds, axis=1), y_test[0:num_images])
    fgsm = accuracy.result()
    print("ACC: ", accuracy.result())
    """

    accuracy = tf.keras.metrics.Accuracy()
    preds = model.predict(X_test[0:num_images])
    accuracy.update_state(np.argmax(preds, axis=1), y_test[0:num_images])
    res = accuracy.result()
    acc.append(float(res.numpy()))
    print("[%s] Accuracy: "%(w), res)
    accuracy = tf.keras.metrics.Accuracy()
    if attack == 'GA':
        adv = meth(model, X_test[0:num_images], G=args.G, R=args.R, N=args.N, D=0.075)
    else:
        adv = meth(model, X_test[0:num_images], eps=0.075, loss_fn=loss, 
                   num_models=10, order=ord, direction=y_test[0:num_images])#, num_steps=15)
    preds = model.predict(adv)
    accuracy.update_state(np.argmax(preds, axis=1), y_test[0:num_images])
    res = accuracy.result()
    print("[%s] Robustness: "%(w), res)
    results.append(float(res.numpy()))

    accuracy = tf.keras.metrics.Accuracy()
    preds = analyzers.chernoff_bound_verification(model, X_test[0:100], 0.075, y_test[0:100], confidence=0.80)
    accuracy.update_state(np.argmax(preds, axis=1), y_test[0:100])
    print("Massart Lower Bound (IBP): ",  accuracy.result())
    res = accuracy.result()
    veri.append(float(res.numpy()))

print(np.around(acc, 2))
print(np.around(results, 2))
print(np.around(veri, 2))
"""
latex_string_header = "\begin{table}[] \n \begin{tabular}{llllll|l} \n Inference Method & BBB & VOGN & NoisyAdam & SWAG & SWAG-FC & SGD \\ \hline"
latex_format_line = "Attack Name      & %.3f   &  %.3f  &  %.3f   & %.3f   & %.3f   & %.3f   "%(tuple(results))
latex_string_footer = "\end{tabular} \n \end{table}"

print_val = latex_string_header + latex_format_line + latex_string_footer
print(print_val)
print(latex_format_line)
"""
