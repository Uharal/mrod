
import holidays
import pandas as pd
import numpy as np

def get_lifepattern(stay,move):

    stay = stay.sort_values(by = 'rank')
    move.loc[move['etype'].isnull(),'etype']='O_0'
    move = move.sort_values(by = 'rank')

    # 重命名move DataFrame的列，以便与stay DataFrame的列匹配
    move = move.rename(columns={'shour': 'hour'})

    # 创建一个新的列'type'，将'stype'和'etype'合并
    move['type'] = move['stype'] + '.' + move['etype']

    # 从move DataFrame中选择必要的列
    move_lifepattern = move[['reindex', 'type', 'hour', 'count']]

    # 定义一个函数以扩展小时范围
    def expand_hours(row):
        if row['shour'] <= row['ehour']:
            return list(range(row['shour'], row['ehour'] + 1))
        else:
            return list(range(row['shour'], 24)) + list(range(0, row['ehour'] + 1))

    # 应用expand_hours函数并扩展DataFrame
    stay_tag = stay[['type', 'shour', 'ehour']].drop_duplicates()
    expanded_hours = stay_tag.apply(expand_hours, axis=1)

    df_expanded = stay_tag.loc[stay_tag.index.repeat(expanded_hours.str.len())]

    df_expanded['hour'] = [hour for sublist in expanded_hours for hour in sublist]

    # 重置最终DataFrame的索引
    df_expanded.reset_index(drop=True, inplace=True)

    # 合并stay和df_expanded DataFrame，按'reindex'，'type'和'hour'分组并计算总数
    stay_lifepattern = pd.merge(stay, df_expanded).groupby(['reindex', 'type', 'hour'])['count'].sum().reset_index()

    # 修改'type'列的值，将其合并为类型对
    stay_lifepattern['type'] = stay_lifepattern['type'] + '.' + stay_lifepattern['type']

    # 合并move_lifepattern和stay_lifepattern
    lifepattern = pd.concat([move_lifepattern, stay_lifepattern])
    lifepattern['otype'] = lifepattern['type'].apply(lambda x: x.split('.')[0])
    lifepattern['dtype'] = lifepattern['type'].apply(lambda x: x.split('.')[1])
    return lifepattern

def generate_seq(lifepattern_i,days=100,seq_type='df',starttime = '2020-01-01',
                 workday_adjust = 2):
    def getprob(f):
        f = f[['dtype','count']]
        f['prob'] = f['count']/f['count'].sum()
        return f[['dtype','prob']].values
    def p_adjust(p,isworkday):
        p = p.copy()
        if not isworkday:
            workindex = pd.Series(p[:,0]).str.contains('W')
            otherindex = pd.Series(p[:,0]).str.contains('O')
            p[workindex,1]/=workday_adjust
            p[otherindex,1]*=workday_adjust
            p[:,1] = p[:,1]/p[:,1].sum()
        return p
    
    lifepattern_dict = lifepattern_i.groupby(['hour','otype']).apply(lambda x:getprob(x)).to_dict()
    lifepattern_dict_hour = lifepattern_i.groupby(['hour']).apply(lambda x:getprob(x)).to_dict()
    initstate = lifepattern_i.groupby(['otype'])['count'].sum().index[0]
    # 马尔科夫链
    currenthour = 0
    currentstate = initstate
    allstates = [currentstate]
    repeattimes = 0
    for i in range(24*days-1):
        day = i//24
        date = pd.Timestamp(starttime)+pd.Timedelta(days = day)
        #是否为节假日为工作日
        cn_holiday = holidays.CN()
        if (date.dayofweek in [5,6]) | (date in holidays.CN()):
            isworkday = False
        else:
            isworkday = True

        #print(date,isworkday)
        
        if (currenthour,currentstate) in lifepattern_dict:
            p = lifepattern_dict[(currenthour,currentstate)]
            p = p_adjust(p,isworkday)
            
            nextstate = np.random.choice(p[:,0],size = 1,p=list(p[:,1]))[0]
        else:
            #此处为随机选择
            if currenthour in lifepattern_dict_hour:
                p = lifepattern_dict_hour[currenthour]
                p = p_adjust(p,isworkday)
                
                nextstate = np.random.choice(p[:,0],size = 1,p=list(p[:,1]))[0]
            else:
                nextstate = initstate

        currenthour+=1
        if currenthour== 24:
            currenthour = 0

        #重复过多次则剔除
        if nextstate == currentstate:
            
            repeattimes += 1
            if repeattimes == 24:
                nextstate = initstate
                repeattimes = 0
        else:
            repeattimes = 0
        currentstate = nextstate

        allstates.append(currentstate)
    if seq_type == 'matrix':
        #pass
        return np.array(allstates).reshape(-1,24).tolist()

    elif seq_type == 'df':
        allstates = pd.DataFrame(allstates,columns=['type'])
        allstates['hour'] = range(len(allstates))
        allstates['time'] = allstates['hour'].apply(lambda x:pd.Timestamp(starttime)+pd.Timedelta(hours = x))+np.random.uniform(0*60,60*60,len(allstates)).astype(int)*pd.Timedelta('1 second')
        allstates = allstates[(allstates['type'].shift())!=allstates['type']]
        return allstates[['time','type']]

