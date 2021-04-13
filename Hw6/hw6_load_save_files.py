import numpy as np
import pandas as pd
import os
import time
import torch
#%%
query_df_train = pd.read_csv('C:\\Users\\LAB\\Desktop\\IR-hw6-data\\train_queries.csv')
document_df = pd.read_csv('C:\\Users\\LAB\\Desktop\\IR-hw6-data\\documents.csv')
query_df_test = pd.read_csv('C:\\Users\\LAB\\Desktop\\IR-hw6-data\\test_queries.csv')
#%%
test = query_df_train[100:]
#%% 新增bert_df 存放BM25_top1000
def create_train_data(query_df , doc_df):
    topk = 1000
    negative_df = pd.DataFrame(columns = ["query_text","docs_id", "doc_text" ,  "positive"])
    positive_df = pd.DataFrame(columns = ["query_text","docs_id", "doc_text" ,  "positive"])
    all_positive_df = pd.DataFrame(columns = ["query_text","docs_id", "doc_text"])
    #empty dataframe
    start = time.time()
    # 建立資料
    for query_id in range(100):#query_df.shape[0]
        s1 = time.time()
        # initialize
        query_num = np.zeros(topk)
        query_text = []
        positive_query_text = []
        relevant_docs = []
        doc_text =  []
        pos_docs_ids = []
        positive_docs = np.zeros(topk)
        postive_doc_text = []
        
        print(query_id)
        query_num[ : topk] = query_df['query_id'][query_id] #取得query_id
        for i in range(topk):
            query_text.append(query_df['query_text'][query_id])
        relevant_docs = query_df['bm25_top1000'][query_id].split() # bm25 top1000
        pos_docs_ids = query_df['pos_doc_ids'][query_id].split() #positive docs
        
        for i in range(len(pos_docs_ids)):
            positive_query_text.append(query_df['query_text'][query_id])
            
        for index ,doc_id in enumerate(pos_docs_ids):
            postive_doc_text.append(doc_df[doc_df['doc_id'] == doc_id]['doc_text'].tolist())
            
        for index ,doc_id in enumerate(relevant_docs):
            doc_text.append(doc_df[doc_df['doc_id'] == doc_id]['doc_text'].tolist())
            if doc_id in pos_docs_ids:
                positive_docs[index] = 1
            else:
                positive_docs[index] = 0
        # 取得query所有的正文章
        positive_d = {"query_text" : positive_query_text , "docs_id" : pos_docs_ids , "doc_text" : postive_doc_text}
        # 取得query 在 BM25 1000 下的 正負文章
        d = {"query_text" : query_text , "docs_id" :relevant_docs, \
               "doc_text" : doc_text , "positive":positive_docs}
        this_positive_df = pd.DataFrame(data = positive_d)
        this_df = pd.DataFrame(data = d)
        positive_inBM25 = this_df[this_df['positive'] == 1]
        negative_inBM25 = this_df[this_df['positive'] == 0]
        positive_df = pd.concat([positive_df , positive_inBM25] , ignore_index = True)
        negative_df = pd.concat([negative_df , negative_inBM25] , ignore_index = True)
        all_positive_df = pd.concat([all_positive_df , this_positive_df] , ignore_index = True)
        s2 = time.time()
        print('this loop :' + str(s2 - s1) + 's')
        
    end = time.time()
    print('total time :' + str(end - start) + 's')
    return all_positive_df , positive_df , negative_df

all_positive ,positive_docs ,negative_docs  = create_train_data(query_df_train , document_df)
# 所有正文章 , 出現在BM25Top1000的正文章 , BM25負文章
#%%
all_positive.to_csv("C:\\Users\\LAB\\Desktop\\IR-hw6-data\\data\\all_postive_100queries.csv", index=False)
positive_docs.to_csv("C:\\Users\\LAB\\Desktop\\IR-hw6-data\\data\\BM25Top1000_postive_100queries.csv", index=False)
negative_docs.to_csv("C:\\Users\\LAB\\Desktop\\IR-hw6-data\\data\\BM25Top1000_negative_100queries.csv", index=False)
#%% test_df
def create_test_df(query_df , doc_df):
    topk = 1000
    df = pd.DataFrame(columns = ["query_num","query_text","relevant_docs", "doc_text", "BM25_score"])
    #empty dataframe
    start = time.time()
    # 建立資料
    for query_id in range(query_df.shape[0]):
        s1 = time.time()
        # initialize
        query_num = np.zeros(topk)
        query_text = []
        relevant_docs = []
        doc_text =  []
        BM25_score = []
        
        print(query_id)
        query_num[ : topk] = query_df['query_id'][query_id]
        for i in range(topk):
            query_text.append(query_df['query_text'][query_id])
        relevant_docs = query_df['bm25_top1000'][query_id].split()
        BM25_score = query_df['bm25_top1000_scores'][query_id].split()
        BM25_score = list(np.float_(BM25_score))
        for index ,doc_id in enumerate(relevant_docs):
            doc_text.append(doc_df[doc_df['doc_id'] == doc_id]['doc_text'].tolist())
                
        d = {"query_num" : query_num ,"query_text" : query_text , "relevant_docs" :relevant_docs, \
               "doc_text" : doc_text , "BM25_score" : BM25_score}
        temp_df = pd.DataFrame(data = d)
        df = pd.concat([df , temp_df] , ignore_index = True)
        s2 = time.time()
        print('this loop :' + str(s2 - s1) + 's')
        
    end = time.time()
    print('total time :' + str(end - start) + 's')
    return df
#%%
test_df = create_test_df(test , document_df)
#%%
test_df.to_csv("C:\\Users\\LAB\\Desktop\\IR-hw6-data\\data\\test_df.tsv", sep="\t", index=False)
test_df.to_csv("C:\\Users\\LAB\\Desktop\\IR-hw6-data\\data\\test_df.csv", index=False)
