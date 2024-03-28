import pandas as pd
import numpy as np
import geopandas as gpd
import transbigdata as tbd
import warnings
warnings.filterwarnings('ignore')

paramssh = {'slon': 120.88125,
        'slat': 30.7125,
        'deltalon': 0.0125,
        'deltalat': 0.008333,
        'theta': 0,
        'method': 'rect',
        'gridsize': 1000}
def radiation(homesh,worksh,rij):
    # 辐射模型
    # 栅格参数

    homesh = homesh.copy()
    worksh = worksh.copy()
    rij = rij.copy()
    homesh.columns = ['hgrid','hcount']
    worksh.columns = ['wgrid','wcount']
    rij.columns = ['hgrid','wgrid','dis']

    homesh['T_i'] = homesh['hcount']*worksh['wcount'].sum()/homesh['hcount'].sum()
    a = homesh['hgrid'].str.split(',').apply(lambda r:tbd.grid_to_centre([int(r[0]),int(r[1])], paramssh))
    homesh['hlon'] = a.apply(lambda r:r[0])
    homesh['hlat'] = a.apply(lambda r:r[1])

    a = worksh['wgrid'].str.split(',').apply(lambda r:tbd.grid_to_centre([int(r[0]),int(r[1])], paramssh))
    worksh['wlon'] = a.apply(lambda r:r[0])
    worksh['wlat'] = a.apply(lambda r:r[1])

    o = homesh[['hgrid','hlon','hlat','hcount','T_i']].rename(columns={'hcount':'m_i','wcount':'n_i'})
    d = worksh[['wgrid','wlon','wlat','wcount']].rename(columns={'hcount':'m_i','wcount':'n_i'})
    o['flag'] = 1
    d['flag'] = 1
    od = pd.merge(o,d)

    od = pd.merge(od,rij)

    distance_counttable = od.groupby(['hgrid','dis'])['n_i'].sum().reset_index()
    distance_counttable = distance_counttable.pivot('hgrid','dis','n_i').fillna(0)
    distance_counttable = distance_counttable.cumsum(axis = 1).unstack().reset_index().rename(columns={0:'sij'})

    od = pd.merge(od,distance_counttable,how = 'left')
    od['sij'] = od['sij'].fillna(0)
    od['T_ij'] =  od['T_i']*(od['m_i']*od['n_i'])/((od['m_i']+od['sij'])*(od['m_i']+od['n_i']+od['sij']))
    od = od[['hgrid','hlon','hlat','wgrid','wlon','wlat','T_ij']]
    return od



def heatmap_to_jointprob(heatmap_24h,coef,rij):
    # 热力数据估计联合分布
    # 估算每个栅格的home work other count
    print('正在计算每个栅格的home work other count')
    HWOcount = pd.DataFrame(np.dot(heatmap_24h.values,coef),columns=['home','work','other'])
    HWOcount['grid'] = heatmap_24h.index
    W_prob = HWOcount[['grid','work']].rename(columns={'work':'count'})
    H_prob = HWOcount[['grid','home']].rename(columns={'home':'count'})
    O_prob = HWOcount[['grid','other']].rename(columns={'other':'count'})

    W_prob = W_prob[W_prob['count']>0]
    H_prob = H_prob[H_prob['count']>0]
    O_prob = O_prob[O_prob['count']>0]

    # 辐射模型估算联合分布
    print('正在计算HW联合分布')
    HW_prob = radiation(H_prob,W_prob,rij)
    HW_prob = HW_prob[HW_prob['hgrid']!=HW_prob['wgrid']]

    print('正在计算HO联合分布')
    HO_prob = radiation(H_prob,O_prob,rij)
    HO_prob.columns = ['hgrid','hlon','hlat','ogrid','olon','olat','T_ij']
    HO_prob = HO_prob[HO_prob['hgrid']!=HO_prob['ogrid']]

    print('正在计算WO联合分布')
    WO_prob = radiation(W_prob,O_prob,rij)
    WO_prob.columns = ['wgrid','wlon','wlat','ogrid','olon','olat','T_ij']
    WO_prob = WO_prob[WO_prob['wgrid']!=WO_prob['ogrid']]
    return W_prob,H_prob,O_prob,HW_prob,HO_prob,WO_prob


def generate_key_locs(H_prob,HO_prob,HW_prob,WO_prob):


    # 生成关键点位置
    def choose_loc_single(prob_df):
        try:
            return np.random.choice(prob_df['grid'],size=1,p=prob_df['count']/prob_df['count'].sum())[0]
        except:
            return None
    def choose_loc_joint(prob_df):
        try:
            prob_df = prob_df.copy()
            prob_df.columns = ['ogrid','olon','olat','dgrid','dlon','dlat','count']
            return np.random.choice(prob_df['dgrid'],size=1,p=prob_df['count']/prob_df['count'].sum())[0]
        except:
            return None
    #
    H_0 = choose_loc_single(H_prob)

    HOp = HO_prob[HO_prob['hgrid']==H_0]
    HWp = HW_prob[HW_prob['hgrid']==H_0]
    H_1 = choose_loc_joint(HOp)
    H_2 = choose_loc_joint(HOp)
    H_3 = choose_loc_joint(HOp)
    H_4 = choose_loc_joint(HOp)
    W_0 = choose_loc_joint(HWp)
    W_1 = choose_loc_joint(HWp)
    W_2 = choose_loc_joint(HWp)
    W_3 = choose_loc_joint(HWp)
    W_4 = choose_loc_joint(HWp)
    oprob = pd.concat([HOp[['ogrid','T_ij']],
                       WO_prob[WO_prob['wgrid']==W_0][['ogrid','T_ij']]])
    oprob.columns = ['grid','count']
    O_0 = choose_loc_single(oprob)
    O_1 = choose_loc_single(oprob)
    O_2 = choose_loc_single(oprob)
    O_3 = choose_loc_single(oprob)
    O_4 = choose_loc_single(oprob)
    O_5 = choose_loc_single(oprob)
    O_6 = choose_loc_single(oprob)
    O_7 = choose_loc_single(oprob)
    O_8 = choose_loc_single(oprob)
    O_9 = choose_loc_single(oprob)

    key_locs = {
        'H_0':H_0, 'H_1':H_1, 'H_2':H_2, 'H_3':H_3, 'H_4':H_4, 
        'W_0':W_0, 'W_1':W_1, 'W_2':W_2, 'W_3':W_3, 'W_4':W_4,
        'O_0':O_0, 'O_1':O_1, 'O_2':O_2, 'O_3':O_3, 'O_4':O_4,
        'O_5':O_5, 'O_6':O_6, 'O_7':O_7, 'O_8':O_8, 'O_9':O_9
    }
    key_locs = pd.DataFrame([key_locs]).T
    key_locs.columns=['grid']
    key_locs = key_locs[~key_locs['grid'].isnull()]
    key_locs['loncol'] = key_locs['grid'].str.split(',').apply(lambda x:x[0]).astype(int)
    key_locs['latcol'] = key_locs['grid'].str.split(',').apply(lambda x:x[1]).astype(int)
    key_locs['lon'],key_locs['lat'] = tbd.grid_to_centre([key_locs['loncol'],key_locs['latcol']],paramssh)

    # 关键点位置加入随机扰动
    key_locs['lon']+=np.random.uniform(-paramssh['deltalon']/2,paramssh['deltalon']/2,len(key_locs))
    key_locs['lat']+=np.random.uniform(-paramssh['deltalat']/2,paramssh['deltalat']/2,len(key_locs))

    key_locs['lon'] = key_locs['lon'].round(6)
    key_locs['lat'] = key_locs['lat'].round(6)
    import json
    return json.loads(key_locs.apply(lambda x:(x['lat'],x['lon']),axis = 1).to_json())



