import pandas as pd
from model import MyModel
from config import parsers
import torch
from transformers import BertTokenizer
import time
import re

def load_model(device, model_path):
    myModel = MyModel().to(device)
    myModel.load_state_dict(torch.load('model/best_model.pth'))
    myModel.eval()
    return myModel


def process_text(text, bert_pred):
    tokenizer = BertTokenizer.from_pretrained(bert_pred)
    token_id = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokenizer.tokenize(text))
    mask = [1] * len(token_id) + [0] * (args.max_len + 2 - len(token_id))
    token_ids = token_id + [0] * (args.max_len + 2 - len(token_id))
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    mask = torch.tensor(mask).unsqueeze(0)
    x = torch.stack([token_ids, mask])
    return x


def text_class_name(pred):
    result = torch.argmax(pred, dim=1)
    result = result.cpu().numpy().tolist()
    classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)), classification))
    return classification_dict[result[0]]


filepath = 'C:\\Users\\jwt\\Desktop\\微博研究\\数据\\用户分类数据.csv'
print('数据集路径:'+filepath)
df_data = pd.read_csv(filepath)
pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9，。？！；：,.?!;:\n]')
pattern1 = re.compile(r'#[\u4e00-\u9fa5]+?#')
pattern2 = re.compile(r'L.{1,10}的微博视频')
pattern3 = re.compile(r'[展开c]')
pattern4 = re.compile(r'[收起d]')
pattern5 = re.compile(r'<img src=.+?>')
pattern6 = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}')

args = parsers()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = load_model(device, args.save_model_best)

#finance,realty,stocks,education,science,society,politics,sports,game,entertainment,military

pred_list = []
score_list1 = []
score_list2 = []
score_list3 = []
score_list4 = []
score_list5 = []
score_list6 = []
score_list7 = []
score_list8 = []
score_list9 = []
score_list10 = []
score_list11 = []
for index in range(df_data.shape[0]):
    print(index)
    wei_str = str(df_data.loc[index,'微博内容'])
    wei_str = re.sub(pattern4,'',wei_str)
    wei_str = re.sub(pattern3,'',wei_str)
    wei_str = re.sub(pattern2,'',wei_str)
    wei_str = re.sub(pattern1,'',wei_str)
    wei_str = re.sub(pattern5,'',wei_str)
    wei_list = re.split(pattern6,wei_str)
    for w_str in wei_list:
        w_str = re.sub(pattern6,'',w_str)
        w_str = re.sub(pattern,'',w_str)
        x = process_text(w_str, args.bert_pred)
        with torch.no_grad():
            pred = model(x)
        pred_list.append(text_class_name(pred))
    res1 = pred_list.count('military')
    res2 = pred_list.count('politics')
    res3 = pred_list.count('finance')
    res4 = pred_list.count('realty')
    res5 = pred_list.count('stocks')
    res6 = pred_list.count('education')
    res7 = pred_list.count('science')
    res8 = pred_list.count('society')
    res9 = pred_list.count('sports')
    res10 = pred_list.count('game')
    res11 = pred_list.count('entertainment')
    if int(df_data.loc[index, '爬取微博数']) != 0:
        score_list1.append(res1 / int(df_data.loc[index, '爬取微博数']))
        score_list2.append(res2 / int(df_data.loc[index, '爬取微博数']))
        score_list3.append(res3 / int(df_data.loc[index, '爬取微博数']))
        score_list4.append(res4 / int(df_data.loc[index, '爬取微博数']))
        score_list5.append(res5 / int(df_data.loc[index, '爬取微博数']))
        score_list6.append(res6 / int(df_data.loc[index, '爬取微博数']))
        score_list7.append(res7 / int(df_data.loc[index, '爬取微博数']))
        score_list8.append(res8 / int(df_data.loc[index, '爬取微博数']))
        score_list9.append(res9 / int(df_data.loc[index, '爬取微博数']))
        score_list10.append(res10 / int(df_data.loc[index, '爬取微博数']))
        score_list11.append(res11 / int(df_data.loc[index, '爬取微博数']))
    else:
        score_list1.append(0)
        score_list2.append(0)
        score_list3.append(0)
        score_list4.append(0)
        score_list5.append(0)
        score_list6.append(0)
        score_list7.append(0)
        score_list8.append(0)
        score_list9.append(0)
        score_list10.append(0)
        score_list11.append(0)
    pred_list.clear()
df_mi = pd.DataFrame(score_list1,columns=['军事文本占比'])
df_po = pd.DataFrame(score_list2,columns=['政治文本占比'])
df_fi = pd.DataFrame(score_list3,columns=['财经文本占比'])
df_re = pd.DataFrame(score_list4,columns=['房产文本占比'])
df_st = pd.DataFrame(score_list5,columns=['股票文本占比'])
df_ed = pd.DataFrame(score_list6,columns=['教育文本占比'])
df_sc = pd.DataFrame(score_list7,columns=['科学文本占比'])
df_so = pd.DataFrame(score_list8,columns=['社会文本占比'])
df_sp = pd.DataFrame(score_list9,columns=['体育文本占比'])
df_ga = pd.DataFrame(score_list10,columns=['游戏文本占比'])
df_en = pd.DataFrame(score_list11,columns=['娱乐文本占比'])

df_sum = pd.concat([df_data,df_po,df_mi,df_fi,df_re,df_st,df_ed,df_sc,df_so,df_sp,df_ga,df_en],axis=1)
df_sum.to_csv(filepath,index = False,encoding='utf_8_sig')
