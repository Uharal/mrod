#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import geopandas as gpd
import transbigdata as tbd
import warnings
warnings.filterwarnings('ignore')

# 根据共同分布生成关键点位置
from utils.keyloc_generate import generate_key_locs
import utils.TbdGeo as TbdGeo


# In[14]:


#Life pattern
lifepattern = pd.read_csv('Lifepattern/Lifepattern.csv')

#Key location
H_prob = pd.read_csv(r'Keylocation/H_prob.csv')
HW_prob = pd.read_csv(r'Keylocation/HW_prob.csv')
HO_prob = pd.read_csv(r'Keylocation/HO_prob.csv')
WO_prob = pd.read_csv(r'Keylocation/WO_prob.csv')


# In[15]:


def generate_traj(days,seed):
    #设置不同的random seed
    np.random.seed(seed)
    traj = TbdGeo.generate_seq(lifepattern[lifepattern['reindex'] == lifepattern['reindex'].sample(1).iloc[0]],days = days,seq_type = 'df',starttime='2023-07-12')

    key_loc = generate_key_locs(H_prob,HO_prob,HW_prob,WO_prob)
    key_loc = pd.DataFrame(key_loc).T.reset_index()
    key_loc.columns = ['type','lat','lon']

    traj = pd.merge(traj,key_loc,on='type',how='left')
    return traj


# In[16]:


N = 10
days = 7
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)
g = pd.DataFrame(range(N))[0].parallel_apply(lambda x:generate_traj(days,seed = x))
def reid_traj(x):
    uid = x['index']
    traj = x[0]
    traj['uid'] = uid
    return traj
g = pd.concat(list(g.reset_index().apply(lambda x:reid_traj(x),axis=1)))

g.to_csv(f'result/synthetic_data_{N}.csv',index=None)


# In[17]:

#单一条轨迹可视化
#用于mobmap可视化的数据导出

g2 = g.copy()
g2['time'] = pd.to_datetime(g2['time'])
g2['timenext'] =  g2['time'].shift(-1)
g2 = g2.iloc[:-1]
g2['time'] = g2['time'].shift(-1)-1800*pd.Timedelta('1 second')
traj = pd.concat([g,g2.iloc[:-1]]).sort_values('time')
#traj.to_csv(f'result/synthetic_data_{N}_vis.csv',index=None)


uids = traj['uid'].drop_duplicates()

import utils.plot_traj as plot_traj
plot_traj.plot_traj(g[g['uid']==g['uid'].drop_duplicates().sample(1).iloc[0]],
                    starttime='2023-07-12 00:00:00'
                    ,days = 1,fix = 2)





