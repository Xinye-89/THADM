# -*- coding: utf-8 -*-

import networkx as nx
from sklearn.feature_extraction import FeatureHasher

#---heterogeneous network---
def generate_hetero_g(df):
    G=nx.Graph()
    ids=list(set(df['patient_id2']))
    print(len(ids))
    addrs=list(set(df['hospital_id2']))
    print(len(addrs))
    for i in ids:
        G.add_node(i,type='patient_id')
    for i in addrs:
        G.add_node(i,type='hospital')
    for i in range(len(df)):
        G.add_edge(df['patient_id2'].iloc[i],df['hospital_id2'].iloc[i])
    return G

#----hash coding----
def hashFunc(df,hash_people_n=32,hash_addr_n=16):
    # people_id/addr/event_begin_time/event_end_time
    hasher=FeatureHasher(n_features=hash_people_n,input_type='string')
    hash_feats=hasher.transform(df['patient_id'].apply(lambda x:[str(x)]))
    hash1=hash_feats.toarray()
    hasher=FeatureHasher(n_features=hash_addr_n,input_type='string')
    hash_feats=hasher.transform(df['hospital_id'].apply(lambda x:[str(x)]))
    hash2=hash_feats.toarray()
    return hash1,hash2
