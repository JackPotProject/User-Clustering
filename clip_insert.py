import torch
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from PIL import Image
import pandas as pd
import re
from transformers import BertTokenizer, PegasusForConditionalGeneration, Text2TextGenerationPipeline
import urllib.request
import traceback
import numpy as np


# -----------------------------------------------------------------------------------------------#
def model_pre(img_url, save_path, pre_text):
    try:
        urllib.request.urlretrieve(img_url, save_path)
        image = preprocess(Image.open(save_path)).unsqueeze(0).to(device)
        summ = text2text_generator(pre_text, max_length=70, do_sample=False)
        result = summ[0]['generated_text'].replace(" ", "")
        text = clip.tokenize(result).to(device)
        avg_score = np.zeros((1, 1))
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            logits_per_image, logits_per_text = model.get_similarity(image, text)
            probs = logits_per_image.cpu().numpy()
        # print("Label probs:", probs)
        if len(probs) != 0:
            avg_score = probs.mean()
            return avg_score
        else:
            return avg_score
    except Exception as e:
        traceback.print_exc()


def save_result_to_file(result, filename):
    with open(filename, 'a') as file:
        file.write(str(result) + '\n')


# img_url = 'https://wx1.sinaimg.cn/wap180/001Ds5xfly1hjb39jih50j60jg0cy76602.jpg'
# save_path = 'cache.jpg'
# pre_text = '纽约爆发大规模反战游行 大中央车站一度关闭 据路透社报道，当地时间27日晚，数百名犹太和平组织成员聚集在纽约大中央车站，呼吁巴以冲突双方立即停火。据报道，由于示威人数过多，导致作为纽约主要交通枢纽之一的大中央车站一度关闭。示威人群随后从车站向曼哈顿中城进发，大量人员被当地警方'
# print(model_pre(img_url,save_path,pre_text))
tokenizer = BertTokenizer.from_pretrained("pegasus")
model = PegasusForConditionalGeneration.from_pretrained("pegasus")
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)
filepath = '../数据/用户分类数据(drop).csv'
df_ma = pd.read_csv(filepath)
df_data = df_ma.loc[:, '微博内容']
df_data.dropna(how='微博内容', inplace=False)

pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9，。？！；：,.?!;:\n=<>、 “”【】（）/]')
pattern1 = re.compile(r'#[\u4e00-\u9fa5]+?#')
pattern2 = re.compile(r'L.{1,10}的微博视频')
pattern3 = re.compile(r'展开c')
pattern4 = re.compile(r'收起d')
pattern8 = re.compile(r'...全文')

pattern7 = re.compile(r'<img src=.*?>')
pattern6 = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}')
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./')
model.eval()
sim_list = []  # 存储每条 probs 列表的均值
sim_list_all = []  # 存储每条 sim_list 列表的均值

for index in range(255, df_data.shape[0]):
    print('index=', index)
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
                content = re.sub(pattern, '', content)
                match = re.search(pattern7, content)
                if match:
                    img_url = content[match.start() + 9:]
                    pre_text = content[:match.start()]
                    img_url = re.sub(r'[> ]', '', img_url)
                    if len(img_url) > 10:
                        save_path = 'cache.jpg'
                        avg_scores = model_pre(img_url, save_path, pre_text)
                        if avg_scores:
                            if np.isnan(avg_scores):
                                continue
                            else:
                                sim_list.append(avg_scores)
        if len(sim_list) != 0:
            avg_score_user = np.mean(sim_list)
            print(avg_score_user)
            save_result_to_file(f'{index}   {avg_score_user}', 'clip preserve.txt')
            sim_list_all.append(avg_score_user)
            sim_list.clear()
        else:
            sim_list_all.append(0)
            save_result_to_file(f'{index}   0', 'clip preserve.txt')
    else:
        sim_list_all.append(0)
        save_result_to_file(f'{index}   0', 'clip preserve.txt')
        print(sim_list_all)
        if len(sim_list_all) != index + 1:
            print('出问题的行', index)
df_sim = pd.DataFrame(sim_list_all, columns=['图文一致性'])
df_new = pd.concat([df_ma, df_sim], axis=1)
df_new.to_csv(filepath, index=False, encoding='utf_8_sig')
