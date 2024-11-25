# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from collections import Counter
import dgl
import torch
import random
import networkx as nx
import plotly.graph_objects as go
import random
import plotly.io as pio
from pyvis.network import Network
import matplotlib.pyplot as plt
pio.renderers.default='notebook'
from matplotlib.font_manager import FontProperties
import seaborn as sns
from scipy.stats import chi2_contingency
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt  
from matplotlib.font_manager import FontProperties
import seaborn as sns
from scipy.stats import chi2_contingency
from matplotlib import font_manager as fm

# structural drawing of the medical trajectory network
def network_plot(df_org):
    df=df_org.copy()
    df=df[['patient_id','hospital_id','label']].drop_duplicates()
    du_lst=df['patient_id'].duplicated()
    id_lst=df['patient_id'][du_lst].tolist()
    df=df[df['patient_id'].isin(id_lst)]
    Counter(du_lst)
    var_list=['patient_id','hospital_id']
    for col in var_list:
        df[col]=pd.factorize(df[col])[0]
    df['hospital_id']=df['hospital_id']+max(df['patient_id'])+1
    id_node=list(set(df['patient_id'].tolist()))
    addr_node=list(set(df['hospital_id'].tolist()))
    nodes=id_node+addr_node
    nodes_type=['t1']*len(id_node)+['t2']*len(addr_node)
    tmp=df[['patient_id','hospital_id','label']].drop_duplicates()
    edges_type=tmp['label'].tolist()
    edge_df=tmp[['patient_id','hospital_id']]
    def row_to_tuples(row): 
        return [(row['patient_id'], row[col]) for col in edge_df.columns if col != 'patient_id']
    edges=edge_df.apply(row_to_tuples,axis=1).tolist()
    edges=[item for sublist in edges for item in sublist]
    edges = [(int(edge[0]), int(edge[1])) for edge in edges]  
    
    G=nx.DiGraph()
    for node,node_type in zip(nodes,nodes_type):
        G.add_node(node,type=node_type)
    for edge,edge_type in zip(edges,edges_type):
        G.add_edge(edge[0],edge[1],type=edge_type)
        
    a=[type(edge[1]) for edge in edges]
    Counter(a)
    color_map = {'t1': '#90EE90', 't2': '#0000FF'}
    size_map = {'t1': 15, 't2': 25}
    edge_color_map={0:'black',1:'red'}
    edge_width_map={0:2,1:12}
    colors = [color_map[G.nodes[n]['type']] for n in G.nodes()]
    sizes = [size_map[G.nodes[n]['type']] for n in G.nodes()]
    # pyvis network
    net = Network(notebook=True)
    net.from_nx(G)
    for node in net.nodes:
        node['color'] = color_map[node['type']]
        node['size'] = size_map[node['type']]
    for edge in net.edges:
        edge['color']=edge_color_map[edge['type']]
        edge['width']=edge_width_map[edge['type']]
    # save HTML
    net.show('./Plot/g_plot.html')
    print('plot success.')
    
    
# hyperparameter analysis
np.random.seed(42)   
x = [[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01],
     [2,3,4,5,6,7,8,9,10],
     [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01],
     [2,3,4,5,6,7,8,9,10]] 
  
y = [[82.43,82.43]+[83.11]*8,
     [92.58,92.58,92.88,93.18,93.18]+[93.77]*5,
     [92.63]*3+[93.68]*7,
     [96.93]*10,
     [88.66]*2+[89.18]*8,
     [88.87,90.46]+[91.05]*3+[92.25]*3+[92.64]*2,
    
     [83.11,62.16,12.84,02.03]+[00.00]*5,
     [93.18,70.33,51.93,21.07,16.02,06.23,03.56,00.89,00.89],
     [93.68,80.00,44.21,15.26,11.05,07.89,01.58,00.00,00.00],
     [96.93,88.34,55.21,32.52,28.22,20.86,11.66,00.00,00.00],
     [89.18,74.74,44.85,15.98,06.19,01.03,01.03,00.00,00.00],
     [91.05,56.26,27.63,04.77,04.77]+[00.00]*4,
     
     [78.79,78.75,79.07,79.06,79.03,79.02,79.02,78.98,78.97,78.96],
     [79.67,79.38,79.28,79.18,78.99,79.21,79.15,79.14,79.12,79.09],
     [80.53,80.44,80.36,80.85,80.78,80.73,80.67,80.58,80.56,80.54],
     [85.27,84.89,84.63,84.47,84.32,84.21,84.17,84.14,84.11,84.11],
     [77.07,77.03,77.28,77.24,77.23,77.23,77.20,77.16,77.16,77.15],
     [79.41,80.13,80.37,80.32,80.22,80.80,80.76,80.71,80.89,80.85],
     
     [79.06,78.16,55.31,50.67,49.80,49.89,50.00,50.00,50.00],
     [79.18,81.88,74.45,59.82,57.55,52.89,51.64,50.42,50.42],
     [80.85,86.32,70.84,57.19,55.33,53.84,50.76,50.00,50.00],
     [84.47,90.27,76.00,65.47,63.80,60.27,55.74,49.93,49.96],
     [77.24,83.98,71.28,57.69,53.04,50.47,50.49,50.00,50.00],
     [80.32,75.05,62.99,52.22,52.31,49.98,50.00,50.00,50.00]]
  
colors = ['red', 'blue', 'green', 'orange', 'purple', 'grey']  
shapes = ['o', '^', 's','*','x','+']  
labels = ['MICD-1', 'MICD-2', 'MICD-3','MICD-4', 'MICD-5', 'MICD-6']  

font_path = 'C:/Windows/Fonts/simsun.ttc'  
font_prop = fm.FontProperties(fname=font_path,size=14)
plt.rcParams['font.sans-serif'] = ['SimSun'] 
plt.rcParams['axes.unicode_minus'] = False 
  
x_labels=['the neighborhood radius (*)','the minimum number of neighbor nodes (*)',
          'the neighborhood radius (*)','the minimum number of neighbor nodes (*)']
y_labels=['Recall (%)','Recall (%)','AUC (%)','AUC (%)']
top=[113,100,93,93]
fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(12,8))
ax=ax.flatten()
for j in range(4):   
    for i in range(num_groups):  
        print('j:',j,', i:',i)
        line, = ax[j].plot(x[j], y[i+j*6], label=labels[i], color=colors[i], alpha=0.7)   
        ax[j].scatter(x[j], y[i+j*6], color=colors[i], alpha=0.7, marker=shapes[i],s=20)    
    ax[j].legend(handles=[line for line in ax[j].lines[:num_groups]], loc='upper right')    
    ax[j].set_xlabel(x_labels[j],fontproperties=font_prop)    
    ax[j].set_ylabel(y_labels[j],fontproperties=font_prop)    
    ax[j].set_ylim(top=top[j])
plt.show()
    

# Ablation study
import numpy as np  
import matplotlib.pyplot as plt  
from matplotlib.font_manager import FontProperties
import seaborn as sns
from scipy.stats import chi2_contingency
from matplotlib import font_manager as fm

categories = ['MICD-1', 'MICD-2', 'MICD-3','MICD-4', 'MICD-5', 'MICD-6']
sub_categories = ['w/o Node Embedding', 'w/o Hash Coding', 'mtnFID']
num_groups = len(categories)
num_sub_categories = len(sub_categories)
y_labels=['Recall (%)','AUC (%)']

data = [np.random.randint(0, 100, size=(num_groups, num_sub_categories)) for _ in range(2)]
data=[np.array([[78.18,73.23,83.11],[69.21,73.33,93.77],[76.89,74.45,93.68],[87.66,83.21,96.93],[70.33,79.11,89.18],[81.23,79.45,92.64]]),
      np.array([[63.21,68.49,79.07],[54.32,60.28,79.21],[68.33,60.67,80.85],[69.55,68.41,85.27],[67.29,64.48,77.24],[69.23,62.15,80.89]]),]
font_path = 'C:/Windows/Fonts/simsun.ttc'  
font_prop = fm.FontProperties(fname=font_path,size=14)
plt.rcParams['font.sans-serif'] = ['SimSun']  
plt.rcParams['axes.unicode_minus'] = False 
def plot_grouped_bar(ax, data, categories, sub_categories, bar_width=0.15,alpha=0.7,y_label=''):
    index = np.arange(len(categories))
    for i, sub_cat in enumerate(sub_categories):
        bar = ax.bar(index + bar_width * i, data[:, i], bar_width, label=sub_cat,alpha=alpha)
    ax.set_ylabel(y_label,fontproperties=font_prop,fontsize=16)
    ax.set_xticks(index + bar_width * (num_sub_categories - 1) / 2 - bar_width / 2)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=16,ncol=3,loc='upper right')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim(top=130)
fig, axes = plt.subplots(2, 1, figsize=(12, 7))
for i, ax in enumerate(axes):
    print(i)
    plot_grouped_bar(ax, data[i], categories, sub_categories,alpha=0.8,y_label=y_labels[i])
plt.tight_layout()
plt.show()





















