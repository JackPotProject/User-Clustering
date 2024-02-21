from HiFi_Net import HiFi_Net
from PIL import Image
import numpy as np
import torch
import pandas as pd
import re
import urllib.request
import traceback

def model_pre(img_url,save_path):
    prob3 = np.nan
    try:
        urllib.request.urlretrieve(img_url, save_path)
        _, prob3 = HiFi.detect(save_path)
        print("Label probs:", prob3)
    except Exception as e:
        traceback.print_exc()
    return prob3


filepath = 'D:\Project practice\微博研究\数据\用户分类数据.csv'
df_ma = pd.read_csv(filepath)
df_data = df_ma.loc[:,'微博内容']
df_data.dropna(how = '微博内容',inplace = False)

HiFi = HiFi_Net()

pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9，。？！；：,.?!;:\n=<>、 “”【】（）/]')
pattern1 = re.compile(r'#[\u4e00-\u9fa5]+?#')
pattern2 = re.compile(r'L.{1,10}的微博视频')
pattern3 = re.compile(r'展开c')
pattern4 = re.compile(r'收起d')
pattern8 = re.compile(r'...全文')

pattern7 = re.compile(r'<img src=.*?>')
pattern6 = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}')

sim_list = []  # 存储每条 probs 列表的均值
sim_list_all = []  # 存储每条 sim_list 列表的均值

for index in range(df_data.shape[0]):
    print('index=',index)
    wei_str = str(df_data.loc[index])
    if wei_str != 'nan':
        wei_str = re.sub(pattern4, '', wei_str)
        wei_str = re.sub(pattern3, '', wei_str)
        wei_str = re.sub(pattern2, '', wei_str)
        wei_str = re.sub(pattern1, '', wei_str)
        wei_str = re.sub(pattern8, '', wei_str)
        wei_list = re.split(pattern6, wei_str)
        for content in wei_list:
            if content:
                match = re.search(pattern7, content)
                img_url = content[match.start() + 9:]
                img_url = re.sub(r'[> ]', '', img_url)
                if len(img_url) > 10:
                    save_path = 'cache.jpg'
                    avg_score = model_pre(img_url, save_path)
                    if np.isnan(avg_score):
                        continue
                    else:
                        #print(avg_score)
                        sim_list.append(avg_score)
        if len(sim_list) != 0:
            avg_score_user = np.mean(sim_list)
            print(avg_score_user)
            sim_list_all.append(avg_score_user)
            print(sim_list_all)
            sim_list.clear()
        else:
            sim_list_all.append(0)
            print(sim_list_all)
    else:
        sim_list_all.append(0)
        print(sim_list_all)
        if len(sim_list_all) != index+1:
            print('出问题的行',index)

df_sim = pd.DataFrame(sim_list_all,columns=['合成图片得分'])
df_new = pd.concat([df_ma,df_sim],axis = 1)
df_new.to_csv(filepath,index = False,encoding = 'utf_8_sig')

