import pandas as pd
import numpy as np
import re
from datetime import datetime
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from d2l import torch as d2l
import torch.nn.init as init
file_path = '../数据/用户分类数据part1.csv'
df_ma = pd.read_csv(file_path)
#df_ma.drop_duplicates(subset='用户id', keep='first', inplace=True, ignore_index=True)
df_data = df_ma.loc[562:,'微博内容']
#df_data.dropna(how = '微博内容',inplace = False)
def switch_time(year,month,day,hour,mins):
    ti = year + '/' + month + '/' + day + ' ' + hour + ':' + mins
    return ti
data_pattern = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}')
time_list = []
for i in range(df_data.shape[0]):
    time_list.append(data_pattern.findall(str(df_data.iloc[i])))
period = np.zeros((df_data.shape[0],107))
t = []
#list.append的实质是引用对象而不是拷贝对象,所以当被引用的值发生改变时,当前列表也会发生改变
for i in range(len(time_list)):
    for j in range(len(time_list[i])-1):
        t1 = datetime.strptime(time_list[i][j], r"%Y-%m-%d %H:%M")
        t2 = datetime.strptime(time_list[i][j+1], r"%Y-%m-%d %H:%M")
        diff = t1 - t2
        sec = diff.total_seconds()/60
        sec = int(sec)
        if sec >= 2880:
            sec = 2880
        t.append(sec)
    if len(t)>107:
        t = t[:107]
    elif len(t)<107:
        while(len(t)<107):
            t.append(0)
    period[i,:] = t.copy()
    t.clear()
period = np.abs(period)
train_period = []
#如果全空就不加入训练
for i in range(period.shape[0]):
    if not all(period[i]==period[i][0]):
        train_period.append(period[i])

#数据呈现右偏
for i in range(2):
    period_norm = np.log1p(period)
    train_period = np.log1p(train_period)
    train_data = torch.tensor(train_period,dtype = torch.float32)
print(np.max(train_period) , np.min(train_period))
#train_period.shape,train_period[1]


class Encoder(nn.Module):
    def __init__(self,**kwargs):
        super(Encoder,self).__init__()
    def forward(self,x) -> torch.Tensor:
        raise NotImplementedError


class Seq2SeqEncoder(Encoder):
    def __init__(self,input_size,num_hiddens,num_layers,dropout = 0,**kwargs):
        super(Seq2SeqEncoder,self).__init__()
        self.rnn = nn.GRU(input_size,num_hiddens,num_layers,dropout = dropout)
        for layer in [self.rnn]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param.data)
    def forward(self,x):
        x = x.permute((1,0))
        x = x.unsqueeze(2)
        output,state = self.rnn(x)
        return output,state

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs):
        raise NotImplementedError

    def forward(self, X, state) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class Seq2SeqDecoder(Decoder):
    def __init__(self,input_size,num_hiddens,num_layers,dropout = 0,**kwargs):
        super(Seq2SeqDecoder,self).__init__(**kwargs)
        self.rnn = nn.GRU(input_size + num_hiddens,num_hiddens,num_layers,batch_first = False,dropout = dropout)
        self.dense = nn.Linear(num_hiddens,107)
        for layer in [self.rnn,self.dense]:
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param.data)
    def init_state(self,enc_outputs):
        return enc_outputs[1]
    def forward(self,x,state):
        x = x.permute((1,0))
        x = x.unsqueeze(2)
        #state[-1]最后一个时刻的最后一层的输出
        #这里的x.shape[0]是时间步
        #state[-1].repeat((x.shape[0], 1, 1)) 意味着将 state[-1] 这个张量在第一个维度（通常是 batch 维度）上复制 x.shape[0] 次，而在其他维度上不进行复制。
        #经过这个操作就让context与x的形状保持一致
        context = state[-1].repeat((x.shape[0],1,1))
        #从第二个维度叠加,shape = [A,B,C+D]
        concat_context = torch.cat((context,x),2)
        output,state = self.rnn(concat_context,state)
        #再把batch_size提到第零维
        output = self.dense(output).permute(1,0,2)
        return output,state

decoder = Seq2SeqDecoder(input_size = 1,num_hiddens = 10,num_layers = 1)
decoder.eval()
state = decoder.init_state(encoder(x))
output,state = decoder(x,state)
output.shape,state.shape
#(batch_size,num_steps,vocab_size),(num_layers,batch_size,num_hiddens)
class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,**kwargs):
        super(EncoderDecoder,self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    def forward(self,enc_x,dec_x):
        enc_outputs = self.encoder(enc_x)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_x,dec_state)

def train_seq2seq(net,data_iter,lr,num_epochs,device):
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr = lr,weight_decay = 1e-1)
    loss = nn.CrossEntropyLoss()
    net.train()
    for epoch in range(num_epochs):
        #batch的尺寸是(batch_size,time_step)
        for X in data_iter:
            x = X.to(device)
            y = x.to(device)
            dec_input = torch.cat((torch.zeros(y.size(0), 1).long().to(device), y[:, :-1]), dim=1)
            y = y.unsqueeze(2).expand(-1, -1, y.shape[1])
            y_hat,_ = net(x,dec_input)
            l = loss(y_hat,y)
            l.sum().backward()
            #d2l.grad_clipping(net,1)
            optimizer.step()
        if epoch % 100 == 0 :
            print(f'epoch:{int((epoch+100)/100)}/{int(num_epochs/100)},loss:{l}')
input_size,num_hiddens,num_layers,dropout = 1,128,4,0.3
batch_size ,num_steps = 50,25
lr,num_epochs = 0.005,1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_iter = DataLoader(train_data,batch_size,shuffle=True)
encoder = Seq2SeqEncoder(input_size,num_hiddens,num_layers,dropout)
decoder = Seq2SeqDecoder(input_size,num_hiddens,num_layers,dropout)
model = EncoderDecoder(encoder,decoder)
train_seq2seq(model,train_iter,lr,num_epochs,device)

period_data = torch.tensor(period_norm,dtype = torch.float32)
loss = nn.CrossEntropyLoss()
loss_list = []
model.eval()
with torch.no_grad():
    for x in period_data:
        x = x.unsqueeze(0)
        x = x.to(device)
        y = x.to(device)
        #print(y.shape)
        dec_input = torch.cat((torch.zeros(y.size(0), 1).long().to(device), y[:, :-1]), dim=1)
        y = y.unsqueeze(2).expand(-1, -1, y.shape[1])
        reconstruct,_ = model(x,dec_input)
        l = loss(reconstruct, y)
        loss_list.append(l.item())
        # threshold = 10  # 设置阈值，超过阈值则判断为异常
        # if l.item() > threshold:
        #     print('样本 {} 是异常数据，异常概率为 {:.9f}'.format(i+1, l.item()))
        # else:
        #     print('样本 {} 是正常数据，异常概率为 {:.9f}'.format(i+1, l.item()))

len(loss_list)==len(df_ma)

df_score = pd.DataFrame(loss_list,columns=['时间重构误差'])
df_new = pd.concat([df_ma,df_score],axis = 1)
df_new.to_csv(file_path,index = False ,encoding='utf_8_sig')