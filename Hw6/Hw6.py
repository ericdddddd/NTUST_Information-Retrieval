import numpy as np
import pandas as pd
import os
import time
import torch
#%%
from transformers import BertTokenizer

model_version = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_version)
encoded_input = tokenizer("How old are you?", "I'm 6 years old")
print(encoded_input["input_ids"])
#%%
from torch.utils.data import Dataset
 
class BertDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "test"]  # 一般訓練你會需要 dev set
        self.path = "C:\\Users\\LAB\\Desktop\\IR-hw6-data\\data\\"
        self.mode = mode
        # 大數據你會需要用 iterator=True
        if mode == 'train' :
            self.positive_df = pd.read_csv(self.path + "BM25Top1000_postive.csv").fillna("")
            self.negative_df = pd.read_csv(self.path + "BM25Top1000_negative.csv").fillna("")
            self.len = len(self.positive_df)
            self.data = 4
        else:
            self.test_df = pd.read_csv(self.path + "test_df.csv").fillna("")
            self.len = len(self.test_df)
            self.data = 1
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer
    
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        if self.mode == "test":
            text_query = self.test_df.iloc[idx, 1]
            text_doc = self.test_df.iloc[idx, 3]
            label_tensor = None
        else:
            positive_query = self.positive_df.iloc[idx, 0]
            positive_docs = self.positive_df.iloc[idx, 2]
            #隨機從negative中挑三篇出來
            random_docs = np.random.randint(self.negative_df.shape[0], size=3)
            negative_query = self.negative_df.iloc[random_docs,0].values
            negative_docs = self.negative_df.iloc[random_docs,2].values
            
            # 將 label 文字也轉換成索引方便轉換成 tensor
            positive_doc_insert = idx % 4
            label_tensor = torch.tensor(positive_doc_insert).unsqueeze(0)
            label_tensor = label_tensor.type(torch.LongTensor)
            
            positive_doc = [positive_query,positive_docs]
            negative_doc1 = [negative_query[0],negative_docs[0]]
            negative_doc2 = [negative_query[1],negative_docs[1]]
            negative_doc3 = [negative_query[2],negative_docs[2]]
            bert_input = [negative_doc1,negative_doc2,negative_doc3]
            bert_input.insert(positive_doc_insert,positive_doc)
        
        # test , training 時所需要的資料量不一樣大
        input_ids = torch.zeros([self.data, 512], dtype=torch.long)
        token_type_ids = torch.zeros([self.data, 512], dtype=torch.long)
        attention_mask = torch.zeros([self.data, 512], dtype=torch.long)
        
        if self.mode == 'train':
            encoded_input = tokenizer(bert_input , truncation ='longest_first',return_tensors="pt" ,padding = True)
        else:
            encoded_input = tokenizer(text_query,text_doc, truncation ='longest_first',return_tensors='pt', padding=True)
        
        bert_input_shape = list(encoded_input['input_ids'].size())
        word_size = bert_input_shape[1]
        
        input_ids[:,:word_size] = encoded_input['input_ids']
        token_type_ids[:,:word_size] = encoded_input['token_type_ids']
        attention_mask[:,:word_size] = encoded_input['attention_mask']
        
        #return (encoded_input['input_ids'] ,encoded_input['token_type_ids'] ,encoded_input['attention_mask'], label_tensor)
        return (input_ids,token_type_ids,attention_mask,label_tensor)
    
    def __len__(self):
        return self.len
    
    
# 初始化一個專門讀取訓練樣本的 Dataset，使用中文 BERT 斷詞
train_data = BertDataset("train", tokenizer=tokenizer)
#%%
data = train_data[2]
print(data[0].size())
#data1 = train_data[1]
#print(tokenizer.convert_ids_to_tokens(data[0][0]))
#print(tokenizer.convert_ids_to_tokens(data[0][1]))
#%%
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def train_val_dataset(dataset, val_split=0.1):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

dataset = train_val_dataset(train_data)
trainset = dataset['train']
print(len(trainset))
val_set = dataset['val']
print(len(val_set))
testset =  BertDataset("test", tokenizer=tokenizer)
print(len(testset))
#%%
from torch.utils.data import DataLoader
BATCH_SIZE = 2
testloader = DataLoader(testset, batch_size = BATCH_SIZE,drop_last=True,num_workers=2)
trainloader = DataLoader(trainset, batch_size = BATCH_SIZE,drop_last=True,num_workers=2,shuffle = True)
validationloader = DataLoader(val_set, batch_size = BATCH_SIZE,drop_last=True,num_workers=2,shuffle = True)
#%%
data = next(iter(trainloader))

tokens_tensors = data[0]
segments_tensors = data[1]
masks_tensors = data[2]
label_ids = data[3]
print(f"""
tokens_tensors.shape   = {tokens_tensors.shape} 
{tokens_tensors}
------------------------
segments_tensors.shape = {segments_tensors.shape}
{segments_tensors}
------------------------
masks_tensors.shape    = {masks_tensors.shape}
{masks_tensors}
------------------------
label_ids.shape        = {label_ids.shape}
{label_ids}
""")
#%%
print(label_ids.squeeze())
#%%
test = torch.tensor(0).unsqueeze(0)
print(test)
#%%
from transformers import BertForMultipleChoice

PRETRAINED_MODEL_NAME = "bert-base-cased"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
print(torch.cuda.get_device_name(0))

model = BertForMultipleChoice.from_pretrained(
    PRETRAINED_MODEL_NAME)

model = model.to(device)
# high-level 顯示此模型裡的 modules
print("""
name            module
----------------------""")
for name, module in model.named_children():
    if name == "bert":
        for n, _ in module.named_children():
            print(f"{name}:{n}")
    else:
        print("{:15} {}".format(name, module))
print(model.config)
#%%
temp = tokens_tensors.cuda()
print(type(temp))
#%%
# 訓練模式
model.train(mode = True)

# 使用 Adam Optim 更新整個分類模型的參數
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

EPOCHS = 2
start = time.time()
for epoch in range(EPOCHS):
    s1 = time.time()
    running_loss = 0.0
    for batch_num ,data in enumerate(trainloader):
        
        print('now batch_num : ' + str(batch_num))
        """
        tokens_tensors, segments_tensors, \
        masks_tensors, labels = [t.to(device) for t in data]
        """
        tokens_tensors = data[0].to(device)
        segments_tensors = data[1].to(device)
        masks_tensors = data[2].to(device)
        labels = data[3].squeeze().to(device)
        
        # 將參數梯度歸零
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(input_ids=tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors, 
                        labels=labels)

        loss = outputs.loss
        # backward
        loss.backward()
        optimizer.step()
        
        # 紀錄當前 batch loss
        running_loss += loss.item()
        
    print('[epoch %d] loss: %.3f' %
          (epoch + 1, running_loss))
    s2 = time.time()
    print('this epoch costs :' +  str((s2 - s1) / 60) + 'mins')
    
end = time.time()
print('total time :' +  str((end - start) / 60) + 'mins')
#%%
"""
定義一個可以針對特定 DataLoader 取得模型預測結果以及分類準確度的函式
之後也可以用來生成上傳到 Kaggle 競賽的預測結果

2019/11/22 更新：在將 `tokens`、`segments_tensors` 等 tensors
丟入模型時，強力建議指定每個 tensor 對應的參數名稱，以避免 HuggingFace
更新 repo 程式碼並改變參數順序時影響到我們的結果。
"""
model.eval()

def get_predictions(model, dataloader):
    
    score = None
    with torch.no_grad():
        # 遍巡整個資料集
        for batch_num ,data in enumerate(dataloader):
            # 將所有 tensors 移到 GPU 上
            if batch_num % 100 == 0:
                print('now batch :' + str(batch_num))
            if batch_num > 300:
                break
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            # 別忘記前 3 個 tensors 分別為 tokens, segments 以及 masks
            # 且強烈建議在將這些 tensors 丟入 `model` 時指定對應的參數名稱
            tokens_tensors = data[0]
            segments_tensors = data[1]
            masks_tensors = data[2]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs.logits
            logits = logits.detach().cpu()
            logits = logits[:,1].numpy()
            if score is None:
                score = logits
            else:
                score = np.concatenate((score, logits), axis=None)
            
    return score
    
# 讓模型跑在 GPU 上並取得訓練集的分類準確率
score = get_predictions(model, testloader)

#%%
print(score)
#print(score[0][:,1])
#print(score[0][:,1].numpy())
#%%
test_data = pd.read_csv('C:\\Users\\LAB\\Desktop\\IR-hw6-data\\data\\test_df.csv')
result_df = pd.DataFrame(columns = ["query_id","ranked_doc_ids"])
#%%
score = np.zeros(1000)
topk = 1000
for query_num in range(80):
    if query_num > 2:
        break
    res = {}
    relevant_docs = test_data['relevant_docs'][topk * query_num : topk * (query_num + 1)].tolist()
    query_id = test_data['query_num'][topk * query_num : topk * (query_num + 1)].to_numpy()
    query_BM25_score = test_data['BM25_score'][topk * query_num : topk * (query_num + 1)].to_numpy()
    new_score = query_BM25_score + score#[topk * query_num : topk * (query_num + 1)]
    for i,j in zip(relevant_docs,new_score):
        res[i] = j
    res = sorted(res.items(), key=lambda kv: kv[1],reverse = True)