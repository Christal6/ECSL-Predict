import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec
from multiprocessing import Pool
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV,StratifiedShuffleSplit
import pandas as pd
import numpy as np
import random
from sklearn import svm
import matplotlib.pyplot as plt
import os
from scipy import stats
from collections import Counter
import threading
import time
import sys
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score,roc_curve,auc,balanced_accuracy_score,classification_report
from sklearn.svm import SVC
random.seed(42)

os.environ['CUDA_VISIBLE_DEVICES'] = "1,0"
import torch
from cuml.svm import SVC as SVC_gpu



def extract_feature(i,d,w,n):
	file_path = f'../gene_evalue.csv'
	G = nx.Graph()
	interactions = []
	scores = []
	with open(file_path, 'r') as file:
		for line in file.readlines()[1:]:
			protein1 = line.strip().split(',')[0].split('\t')[0]
			protein2 = line.strip().split(',')[0].split('\t')[1]
			score = float(line.strip().split(',')[1])
			# score = 1/score if score != 0 else 1e200
			# print(protein2)
			print(score)
			G.add_edge(protein1, protein2, weight=score)
	# Precompute probabilities and generate walks
	node2vec = Node2Vec(G, dimensions=d, walk_length=w, num_walks=n, workers=4,p=0.5,q=1 )
	model = node2vec.fit(window=10, min_count=1, batch_words=4)
	gene_features = {}
	for gene in G.nodes:
		gene_features[gene] = model.wv[gene]
	df = pd.DataFrame.from_dict(gene_features,orient='index')
	return df

def ddff(i,d,w,n):
	df = extract_feature(i,d,w,n)
	duplicated_rows = df.index.duplicated(keep=False)
	df = df[~duplicated_rows]
	#rank or not
	# df = df.rank(method='max')
	df = (df - df.min()) / (df.max() - df.min())
	return df


