# ECSL-Predict: Prediction of Synthetic Lethality in *Escherichia coli* Based on Graph Embedding
ECSL-Predict is a novel computational approach that combines five features to predict potential SL pairs in *Escherichia coli*.



## Dataset

- po_sample: 3207 SL gene pairs 
- ne_sample: 294,318 gene pairs 
- independent_test_po: 70 SL gene pairs predicted by *i*AF1260

Other parts of the data are downloaded from the  [NCBI]([Home - GEO - NCBI (nih.gov)](https://www.ncbi.nlm.nih.gov/geo/)) and [STRING]([Downloads - STRING functional protein association networks (string-db.org)](https://cn.string-db.org/cgi/download?sessionId=bAOjsUoDSP5M)) .

## Running the code:

```
cd src
python main.py
python extact_embedding.py
```

## Requirements:

```
python 3.9.7
torch 1.12.0
numpy 1.26.4
pandas 1.2.4
scikit-learn 1.3.0
matplotlib 3.9.0
networkx 2.6.3
gensim 4.3.2
```

