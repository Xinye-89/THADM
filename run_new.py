# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from models.feat_engineer import generate_hetero_g,hashFunc
from models.embed import MNE
from models.detector import detector_eval

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)  # 设置为None表示无限制  
pd.set_option('display.max_columns', None)  # 设置为None表示无限制  
pd.set_option('display.width', None) 

#----数据读取-----
df_train=pd.read_csv('./dataset/MICD_6.csv',index_col=False)

#-----------模型训练------------- 
# 构建异构图
le = LabelEncoder()
df_train['patient_id2']=le.fit_transform(df_train['patient_id'])
df_train['hospital_id2']=le.fit_transform(df_train['hospital_id'])+1+max(df_train['patient_id2'])
G=generate_hetero_g(df_train) 
# hash编码
hash1,hash2=hashFunc(df_train)
# embedding
feat_lst=['feat'+str(x) for x in range(43)]
mne = MNE()
feats_con = mne.fit(G, df_train[feat_lst].values, df_train, hash1, hash2)
# detector
thadm_result,thadm_eval = dbscan_detector(feats_con, result['labels'], eps=0.009, min_samples=2)



    
                                                                                                               
       
            
            
            



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    













