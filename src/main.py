import sys

from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import random
import os
import time
import pickle
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
from multiprocessing import Pool
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
random.seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = "2,0"
import torch
from cuml.svm import SVC as SVC_gpu

def po_ne():
	with open('sample/po_sample.txt', 'r') as file:
		train = file.readlines()
	with open('sample/ne_sample.txt', 'r') as file:
		test = file.readlines()
	return train,test

def sdmv_Matrix(lines,nn):
    df = pd.read_csv(f'../{nn}_feats.csv', sep=',', header=0, index_col=0)
    df = (df - df.min()) / (df.max() - df.min())
    print(df.head())
    features = []
    for line in lines:
        each = line.strip().split('\t')
        if each[0] in df.index and each[1] in df.index:
            gene_1 = [i for i in df.loc[each[0], :]]
            gene_2 = [i for i in df.loc[each[1], :]]
            feature = []
            for i1, i2 in zip(gene_1, gene_2):
                s1 = (i1 + i2) / 2
                s2 = abs(i1 - i2)
                feature.append(s1)
                feature.append(s2)
            features.append(feature)
        else:
            print('why')
            pass
    print(np.shape(features))
    print('ok')
    return features

def main(o):
	for _ in range(5):
		torch.cuda.empty_cache()

	start = time.time()
	out_remain_sample = open('result/file1.csv', 'a')
	out_remain_sample.write('sample,pos,neg\n')
	# load dataset
	output = open(f'result/Ecoli_{o}.csv', 'a')
	output.write('index,feats,C,g,train_score,test_score,train_ACC,test_ACC,train_AUC,test_AUC,train_Pre,test_Pre,train_Rec,test_Rec,train_f1,test_f1\n')
	po_s, ne_s = po_ne()
	ne_lines = random.sample(ne_s, 20000)
	print(ne_lines[:10])

	feature_p = sdmv_Matrix(po_s, o)
	feature_n = sdmv_Matrix(ne_lines, o)

	print('ok')
	print(np.shape(feature_p))
	print(np.shape(feature_n))
	out_remain_sample.write(f'{o},{np.shape(feature_p)[0]},{np.shape(feature_n)[0]}\n')



	YY = np.append(np.ones(np.shape(feature_p)[0]),np.zeros(np.shape(feature_n)[0]))
	XX = np.vstack((feature_p,feature_n))

	# transfer
	XX = XX.astype(np.float32)
	YY = YY.astype(np.float32)


	C = {'ppi':32,'exp':16,'bla':10,'kfe':4}
	gamma = {'ppi':0.1,'exp':0.1,'bla':0.1,'kfe':0.03}


	result = []
	k=0
	be = 0
	for c in [C[o]]:
		for g in [gamma[o]]:
			k+=1
			Sco_1 = []
			Sco_2 = []
			AUC_1 = []
			ACC_1 = []
			f1_1 = []
			recall_1 = []
			Pre_1 = []
			AUC_2 = []
			ACC_2 = []
			f1_2 = []
			recall_2 = []
			Pre_2 = []
			Kf =StratifiedKFold(n_splits=5, random_state=2023 , shuffle=True)
			for i, (train_index, test_index) in enumerate(Kf.split(XX, YY)):
				print(i)
				print(train_index)
				train_x = XX[train_index]
				train_y = YY[train_index]
				test_x = XX[test_index]
				test_y = YY[test_index]

				# # Save the sample of 5 partitions
				# a_s_x = np.array(ss)[train_index]
				# e_s_x = np.array(ss)[test_index]
				# print(e_s_x)
				# train_sam = np.vstack((a_s_x,train_y)).T
				# test_sam = np.vstack((e_s_x,test_y)).T
				# print(train_sam.shape)
				# train_sam_df = pd.DataFrame(train_sam,columns=['genes','class'])
				# test_sam_df = pd.DataFrame(test_sam,columns=['genes','class'])
				# train_sam_df.to_csv(f'/home/xhq/colethal/new_combine_pn/differ_class/sample/train_test/split_{i}_train_sample.csv')
				# test_sam_df.to_csv(f'/home/xhq/colethal/new_combine_pn/differ_class/sample/train_test/split_{i}_test_sample.csv')

				#if need to smote
				# counter = Counter(train_y)
				# print(counter)
				# oversample = SMOTE(sampling_strategy=1,random_state=3047)
				# train_x, train_y = oversample.fit_resample(train_x, train_y)
				# counter = Counter(train_y)
				# print(counter)

				clf = Pipeline([("scale", StandardScaler()), ("svc", SVC_gpu(kernel='rbf', class_weight='balanced', probability=True, C=c, gamma=g))])
				clf.fit(train_x, train_y)

				# 或补充得到所有分数
				s1 = clf.score(train_x, train_y)
				s2 = clf.score(test_x, test_y)
				result_1 = clf.decision_function(train_x)
				result_2 = clf.decision_function(test_x)
				result_3 = clf.predict(train_x)
				result_4 = clf.predict(test_x)


				s3 = balanced_accuracy_score(train_y, result_3)
				s4 = balanced_accuracy_score(test_y, result_4)
				# # Auc
				s5 = roc_auc_score(train_y, result_1)
				s6 = roc_auc_score(test_y, result_2)
				# Precision
				s7 = precision_score(train_y, result_3)
				s8 = precision_score(test_y, result_4)
				s9 = recall_score(train_y, result_3)
				s10 = recall_score(test_y, result_4)
				s11 = f1_score(train_y, result_3)
				s12 = f1_score(test_y, result_4)

				# AUC_m.append(s6)
				Sco_1.append(s1)
				ACC_1.append(s3)
				AUC_1.append(s5)
				Pre_1.append(s7)
				recall_1.append(s9)
				f1_1.append(s11)
				Sco_2.append(s2)
				ACC_2.append(s4)
				AUC_2.append(s6)
				Pre_2.append(s8)
				recall_2.append(s10)
				f1_2.append(s12)
				#
				best_svm_gpu = SVC_gpu(kernel='rbf', C=c, gamma=g, class_weight='balanced', probability=True)
				best_svm_gpu.fit(train_x, train_y)
				pickle.dump(best_svm_gpu,open(f'/Ecoli_{o}_ne20000_split{i}_nor.pkl','wb'), protocol=4)
				# output.write(f'{i},{o},{c},{g},{s1},{s2},{s3},{s4},{s5},{s6},{s7},{s8},{s9},{s10},{s11},{s12}\n')
				print(f'{i},{o},{c},{g},{s1},{s2},{s3},{s4},{s5},{s6},{s7},{s8},{s9},{s10},{s11},{s12}\n')

		# temp = [c, g, np.mean(AUC_2)]
		temp = ['mean', k, c, g, np.mean(Sco_1), np.mean(Sco_2), np.mean(ACC_1), np.mean(ACC_2), np.mean(AUC_1),np.mean(AUC_2), np.mean(Pre_1), np.mean(Pre_2), np.mean(recall_1), np.mean(recall_2), np.mean(f1_1),np.mean(f1_2)]
		print('mean', k, c, g, np.mean(AUC_2))
		output.write(f'mean,{o},{c},{g},{np.mean(Sco_1)},{np.mean(Sco_2)},{np.mean(ACC_1)},{np.mean(ACC_2)},{np.mean(AUC_1)},{np.mean(AUC_2)},{np.mean(Pre_1)},{np.mean(Pre_2)},{np.mean(recall_1)},{np.mean(recall_2)},{np.mean(f1_1)},{np.mean(f1_2)}\n')
		result.append(temp)

		output.write(str(best[1])+"\t"+str(best[2])+"\t"+str(best[8])+"\n")
		output.close()
		out_remain_sample.close()
		for _ in range(10):
			torch.cuda.empty_cache()

		end = time.time()
		print(end - start)

if __name__ == '__main__':
	kind = ['ppi', 'exp', 'bla', 'kfe', 'fit'
	pool = Pool(processes=16)
	for i in kind:
		pool.apply_async(main, args=(i,))
	pool.close()
	pool.join()


