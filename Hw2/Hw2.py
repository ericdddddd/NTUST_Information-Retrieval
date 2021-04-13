# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

#%%
# 讀取檔案

queriesPath ='C:\\Users\\User\\Desktop\\IR\\data\\ntust-ir-2020\\queries'
docsPath = 'C:\\Users\\User\\Desktop\\IR\\data\\ntust-ir-2020\\docs'

queriesList = os.listdir(queriesPath)
queriesList.sort()
docsList = os.listdir(docsPath)
docsList.sort()

queriesContext = []
docsContext = []

for query in queriesList:
    path = os.path.join(queriesPath,query)
    f = open(path)
    context = f.read()
    queriesContext.append(context)

for doc in docsList:
    path = os.path.join(docsPath,doc)
    f = open(path)
    context = f.read()
    docsContext.append(context)


#%%

N = 4191   #檔案總數
Q = 50

#計算df
def cal_df(term,df):
    store = np.zeros(len(term))
    i = 0
    num = 0
    for index in term :
        colcount = 0
        for doc in docsContext:
            context = doc.split( )
            if index in context :
                num += 1
        #print(index + ':' + str(num))
        df[i] = num
        num = 0
        i += 1

# 計算tf
def cal_tf(vector,doc_tf,query_tf):

    if query_tf is None:
        rowcount = 0
        for doc in docsContext :
            context = doc.split( )
            colcount = 0
            for index in vector:
                val = context.count(index)
                doc_tf[rowcount][colcount] = val
                colcount += 1
            rowcount += 1
    elif doc_tf is None:
         index = 0
         for i in vector :
             num = query.count(i)
             query_tf[index] = num
             index += 1   

def splitQuery(query):
    word = query.split( )
    res = []
    for i in word:
        if word not in res:
            res.append(i)
    return res

def cal_doc_len():
    doc_len = np.zeros(N)
    avg_len = 0
    index = 0
    for doc in docsContext:
        document = doc.split()
        doc_len[index] = len(document)
        avg_len += len(document)
        index += 1
        
    avg_len = avg_len / N
    return doc_len , avg_len
#%%

# 計算 BM25
def cal_BM25(vector,df,doc_tf,query_tf,doc_len,avg_len):
    # S1、S3、K1、K3皆為參數(S1=K1+1、S3=K3+1)
    k1 = 3
    k3 = 1
    s1 = k1 + 1
    s3 = k3 + 1
    b = 0.75
    
    # Fi_j Fi_q
    fi_j = np.zeros([N,len(vector)])
    fi_q = np.zeros(len(vector))
    for i in range(N):
        if doc_len[i] == 0:
            fi_j[i] = 0
        else :
            fi_j[i] = s1 * doc_tf[i] /(k1 *((1-b) + b*(doc_len[i]/avg_len))+ doc_tf[i])
    fi_q = s3 * query_tf / (k3 + query_tf)
    
    score = np.zeros([N,len(vector)])
    result = np.zeros(N)
    for doc in range(N):
        score[doc] = fi_j[doc] * fi_q * np.log( 1 + (N - df + 0.5) / (df + 0.5))
        result[doc] = np.sum(score[doc,:])
    return result
#%%
index = 0
score = np.zeros([Q,N])
doc_len = np.zeros(N)
doc_len , avg_len = cal_doc_len()

for query in queriesContext:
    print(index)
    wi = splitQuery(query) # 取得此個query的詞
    # 宣告相關變數
    df = np.zeros(len(wi))
    doc_tf = np.zeros([N,len(wi)])
    query_tf = np.zeros(len(wi))
    
    cal_df(wi,df)  # doc的tf及df
    cal_tf(wi,doc_tf,None)# doc的tf
    cal_tf(wi,None,query_tf)# query的tf
    score[index] = cal_BM25(wi,df,doc_tf,query_tf,doc_len,avg_len)
    index += 1 

#%%
res = {}

save_file_name = 'C:\\Users\\User\\Desktop\\IR\\result\\result12.txt'
fp = open(save_file_name, "w")
fp.seek(0)
fp.write("Query,RetrievedDocuments\n")

for loop in range (len(queriesList)):
    write_string = queriesList[loop][0:-4] + ","
    for i,j in zip(docsList,score[loop]):
        res[i[0:-4]] = j
    sorted_x = sorted(res.items(), key=lambda kv: kv[1],reverse = True)
    #print(sorted_x[:15])
    for doc in sorted_x:
        write_string += doc[0] + " "
    write_string += "\n"
    fp.write(write_string)
    res.clear()
    sorted_x.clear()
fp.truncate()
fp.close()
