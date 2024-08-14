import sys
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import random
import os
import time
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import json
random.seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = "2,0"
import torch
from cuml.svm import SVC as SVC_gpu


def evaluate_model(model, model_name, q):
    feature_a = pd.read_csv(f'results/5_split/split_{q}_train_feats.csv', header=0, index_col=0)
    feature_e = pd.read_csv(f'results/5_split/split_{q}_test_feats.csv', header=0, index_col=0)

    train_x = feature_a.loc[:, feature_a.columns != 'class'].values
    train_y = feature_a['class'].values
    test_x = feature_e.loc[:, feature_a.columns != 'class'].values
    test_y = feature_e['class'].values

    train_x = train_x.astype(np.float32)
    train_y = train_y.ravel().astype(int)  # Ensure train_y is a 1D array and of type int
    test_x = test_x.astype(np.float32)
    test_y = test_y.ravel().astype(int)  # Ensure test_y is a 1D array and of type int

    clf = Pipeline([("scale", MinMaxScaler()), (model_name, model)])
    clf.fit(train_x, train_y)

    s1 = clf.score(train_x, train_y)
    s2 = clf.score(test_x, test_y)
    result_1 = clf.predict_proba(train_x)[:, 1]
    result_2 = clf.predict_proba(test_x)[:, 1]
    result_3 = clf.predict(train_x)
    result_4 = clf.predict(test_x)

    s3 = balanced_accuracy_score(train_y, result_3)
    s4 = balanced_accuracy_score(test_y, result_4)
    s5 = roc_auc_score(train_y, result_1)
    s6 = roc_auc_score(test_y, result_2)
    s7 = precision_score(train_y, result_3)
    s8 = precision_score(test_y, result_4)
    s9 = recall_score(train_y, result_3)
    s10 = recall_score(test_y, result_4)
    s11 = f1_score(train_y, result_3)
    s12 = f1_score(test_y, result_4)

    return [q, model_name, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12]

def main():
    for i in range(5):
        torch.cuda.empty_cache()

    start = time.time()
    output = open(f'Ecoli_means_result_para_5feats_minmax_ml.csv', 'a')
    output.write('random,model,train_score,test_score,train_ACC,test_ACC,train_AUC,test_AUC,train_Pre,test_Pre,train_Rec,test_Rec,train_f1,test_f1\n')

    models = [
        ('LogisticRegression', LogisticRegression(C=10, max_iter=100)),
        ('GaussianNB', GaussianNB()),
        ('MLPClassifier', MLPClassifier(hidden_layer_sizes=(50,), learning_rate_init=0.00002)),
        ('SVC', SVC_gpu(kernel='rbf', class_weight='balanced', probability=True, C=31, gamma=1.96e-5))
    ]

    results = []
    for model_name, model in models:
        model_results = Parallel(n_jobs=-1)(delayed(evaluate_model)(model, model_name, q) for q in range(5))
        results.extend(model_results)

    for result in results:
        output.write(
            f'{result[0]},{result[1]},{result[2]},{result[3]},{result[4]},{result[5]},{result[6]},{result[7]},{result[8]},{result[9]},{result[10]},{result[11]},{result[12]},{result[13]}\n')

if __name__ == '__main__':
    main()
