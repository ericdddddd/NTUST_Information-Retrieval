#%%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#%%

import os

queriesPath = 'C:\\Users\\User\\Desktop\\IR\\data\\ntust-ir-2020\\queries'
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


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%
vector = [] # 全部query的不同單字，vector space model 的 Vector
for query in queriesContext:
    words = query.split( )
    print(words)
    for word in words:
        if word not in vector:
            vector.append(word)
#print(vector)

#%%
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
        print(index + ':' + str(num))
        df[i] = num
        num = 0
        i += 1

def cal_doc_tf(vector,tf):
    rowcount = 0
    for doc in docsContext :
        context = doc.split( )
        colcount = 0
        for index in vector:
            val = context.count(index)
            tf[rowcount][colcount] = val
            colcount += 1
        rowcount += 1
        
def cal_Query_tf(query,vector,tf):
    index = 0
    for i in vector :
        num = query.count(i)
        querytf[index] = num
        index += 1

def cal_TermWeight2(doc_tw,query_tw,doc_tf,query_tf,df):
    if query_tw is None :
        for i in range(N):
            for j in range(df.shape[0]):
                if df[j] == 0 or doc_tf[i][j] == 0 :
                    doc_tw[i][j] = 0
                else: 
                    doc_tw[i][j] = ( 1 + doc_tf[i][j]) * np.log10(N/df[j])
    elif doc_tw is None :
        for j in range(df.shape[0]):
            if df[j] == 0 or query_tf[j] == 0:
                query_tw[j] = 0
            else: 
                query_tw[j] = ( 1 + query_tf[j] ) * np.log10(N/df[j])
        #print(np.max(query_tf))
        #print(query_tw)

def cal_cosSimilarity2(doc_tw,query_tw):
    # 取得個別長度 |q| ,|dj|
    doc_length = np.zeros(N)
    sim = np.zeros(N)
    q_length = np.sqrt(np.dot(query_tw,query_tw))
    #print('q_length :' + str(q_length))
    for i in range(N):
        doc_length[i] = np.sqrt(np.dot(doc_tw[i],doc_tw[i]))
    # 取得內積
    for i in range(N):
        innerProduct = np.dot(query_tw,doc_tw[i])
        #print('innerProduct : ' + str(innerProduct))
        if doc_length[i] * q_length != 0:
            #print('doc_length[i] * q_length' + str(doc_length[i] * q_length))
            sim[i] = innerProduct / (doc_length[i] * q_length)
        elif doc_length[i] * q_length == 0 :
            sim[i] = 0
    return sim

#%%
index = 0
N = 4191 #檔案總數
count = 0
sim = np.zeros([len(queriesList),N], dtype = float)
df = np.zeros(len(vector),dtype=int)

cal_df(vector,df)
#%%

index = 0
for query in queriesContext:
    print(index)
    termw = query.split( ) # 取得此個query的詞
    # 宣告相關變數
    doctf = np.zeros([N,len(vector)])
    querytf = np.zeros(len(vector))
    doc_tw = np.zeros([N,len(vector)], dtype = float) # doc term-weight
    query_tw = np.zeros(len(vector), dtype = float)
    
    cal_doc_tf(vector,doctf)  # doc的tf及df
    #print(doctf)
    cal_Query_tf(termw,vector,querytf)# query的tf
    #print(querytf)
    cal_TermWeight2(doc_tw,None,doctf,None,df) # doc的term-weight
    #print(doc_tw)
    cal_TermWeight2(None,query_tw,None,querytf,df) # query的term-weight
    #print(query_tw)
    sim[index] = cal_cosSimilarity2(doc_tw,query_tw)
    #print(sim[index])
    index += 1
    
#%%

res = {}
save_file_name = 'res.txt'
fp = open(save_file_name, "w")
fp.seek(0)
fp.write("Query,RetrievedDocuments2\n")

for loop in range (len(queriesList)):
    write_string = queriesList[loop][0:-4] + ","
    for i,j in zip(docsList,sim[loop]):
        res[i[0:-4]] = j
    sorted_x = sorted(res.items(), key=lambda kv: kv[1],reverse = True)
    print(sorted_x[:15])
    for doc in sorted_x:
        write_string += doc[0] + " "
    write_string += "\n"
    fp.write(write_string)
    res.clear()
    sorted_x.clear()
fp.truncate()
fp.close()
