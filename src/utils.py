#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from Bio import GenBank
from Bio import SeqIO
import re
from sklearn.metrics import jaccard_score
from scipy.stats import pearsonr
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import mutual_info_regression


def clear_gene_v():
	"clear the same gene"
	df = pd.read_csv('data/gene_pair_evalue.csv',header=0)
	df['gene'] = df['gene'].apply(lambda x: '\t'.join(sorted(x.split('\t'))))
	df.drop_duplicates(subset=['gene'], keep='first', inplace=True)
	df.to_csv('data/gene_evalue.csv',header=True,index=False)

def match_gene(item):
    pattern = r'\[locus_tag=([a-zA-Z0-9]+)\]'
    # 遍历每个元素并匹配
    match = re.search(pattern, item)
    if match:
        print(match.group(1))
    return match.group(1)

def get_fitness():
    "get the fitness scores to knn"
    gene = []
    score = []
    with open(f'../Essentiality.txt', 'r') as f:
        lines = f.readlines()[3:]
        for l in lines:
            score.append(l.split('\t\t')[1])
            gene.append(match_gene(l))
    df = pd.DataFrame({'gene':gene,'fitness':score})
    df.to_csv('f../ff1.csv',header=True,index_label=False)

def pearson_correlation_matrix(X):
    n = X.shape[0]
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr_matrix[i, j] = pearsonr(X[i], X[j])[0]
    pearson_corr = corr_matrix
    distance_matrix = 1 - np.abs(pearson_corr)
    return distance_matrix

def count_fre(seq,k,l):
    "get_kmer_frequence"
    feats = []
    colname = []
    for s in range(3):
        ss = seq[s:]
        # print(ss)
        loc = []
        for i in range(1,k+1):
            if i == 1:
                pass
                loc = [ss[q] for q in range(0,len(ss),3)]
                fre,kk = summary_kmers(loc,i)
                feats.extend(fre)
                colname.extend([f'{i}_{s+1}' for i in kk ])
            elif 1 < i < 6:
                m = 1
                while m < i:
                    n = i - m
                    print(m)
                    print(n)
                    loc = []
                    if m == 1:
                        for u in range(l+1):
                            loc = [ss[q:q+m]+ss[q+u+m:q+u+n+m] for q in range(0,len(ss),3) if q+u+n+m <= len(ss)]
                            fre, kk = summary_kmers(loc, i)
                            feats.extend(fre)
                            colname.extend([f'{i}_{s + 1}_{u}' for i in kk])
                    else:
                        # print(m)
                        for u in range(1,l+1):
                            loc = [ss[q:q+m]+ss[q+u+m:q+u+n+m] for q in range(0,len(ss),3) if q+u+n+m <= len(ss)]
                            fre, kk = summary_kmers(loc, i)
                            feats.extend(fre)
                            colname.extend([f'{i}_{s + 1}_{u}' for i in kk])
                    m += 1
            else:
                m = 1
                while m < i:
                    n = i - m
                    print(m)
                    print(n)
                    loc = []
                    if m == 1:
                        for u in range(2):
                            loc = [ss[q:q + m] + ss[q + u + m:q + u + n + m] for q in range(0, len(ss), 3) if
                                   q + u + n + m <= len(ss)]
                            fre, kk = summary_kmers(loc, i)
                            feats.extend(fre)
                            colname.extend([f'{i}_{s + 1}_{u}' for i in kk])
                    else:
                        # print(m)
                        for u in range(1, 2):
                            loc = [ss[q:q + m] + ss[q + u + m:q + u + n + m] for q in range(0, len(ss), 3) if
                                   q + u + n + m <= len(ss)]
                            fre, kk = summary_kmers(loc, i)
                            feats.extend(fre)
                            colname.extend([f'{i}_{s + 1}_{u}' for i in kk])
                    m += 1
    print(len(feats))
    # print(feats)
    print(colname)
    return feats,colname

def count_tpm():
    df0 = pd.read_csv('../gene_sorted_lengths.csv',index_col=None)
    folder_path ='../raw_problem'
    for filename in os.listdir(folder_path):
        df_ = pd.read_csv(os.path.join(folder_path,filename),sep=',')
        df = pd.read_csv(os.path.join(folder_path,filename),sep=',',index_col=0,encoding="gbk")
        df = df[~df.index.duplicated(keep='first')]         #
        # print(df)
        df_gene = df_.loc[:,'Gene']
        df_lengths = pd.merge(df0,df_gene,left_on='Gene',right_on='Gene',how='inner')
        # df_lengths = pd.merge(df0,df,on='index',how='inner')
        df_lengths.set_index('Gene', inplace=True)
        # print(df_lengths)
        # print('------')
        # print(df_lengths['length'])
        df_deno = df.div(df_lengths['length'],axis=0)
        # print(df_deno)
        # print('------')
        total_counts = df_deno.sum()  # 计算总表达计数值
        # print(total_counts)
        # print('------')
        df_tpm = (df) * 1e6 / total_counts  # 计算TPM
        sorted_df = df_tpm.sort_values(by='Gene', ascending=True)
        print(f'{filename}')
        sorted_df.to_csv(f'../{filename}', sep=',', index=True)
    return

def log_deal():
    """log2 deal"""
    folder_path ='express/dealdata'
    for filename in os.listdir(folder_path):
        df = pd.read_csv(os.path.join(folder_path,filename),sep=',',header=0,index_col=0)
        # 将空缺值填充为0
        df.fillna(0, inplace=True)
        df = df.applymap(lambda x: 0 if x <= 1 else np.log2(x))
        # df_pro = df.applymap(lambda x: 0 if x <= 1 else math.log2(x))   #math
        print(df)
        # df=(df-df.min())/(df.max()-df.min())
        # diff_df = pd.concat([df1, df2]).drop_duplicates(keep=First)
        df.dropna(how='all', inplace=True)
        # df = df[df.applymap(str).apply(lambda x: ~x.str.contains('sample.ginkgo')).any(axis=1)]
        df.to_csv(f'express/{filename}', sep=',', index=True)
        return
