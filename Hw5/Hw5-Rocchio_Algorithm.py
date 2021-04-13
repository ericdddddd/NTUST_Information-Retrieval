import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import time
#%%  讀取檔案

queriesPath ='C:\\Users\\User\\Desktop\\NTUST\\IR\\data\\ntust-ir-2020_hw5_new\\queries'
docsPath = 'C:\\Users\\User\\Desktop\\NTUST\\IR\\data\\ntust-ir-2020_hw5_new\\docs'

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
#%% get tf_idf matrix

max_df = 0.95 #比例
min_df =  5  # 至少出現5次
sublinear_tf = True  # 1 + log(tf)

tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df,
                             smooth_idf= True, sublinear_tf=sublinear_tf)
doc_tf_idf = tfidf_vectorizer.fit_transform(docsContext).toarray()
query_vec = tfidf_vectorizer.transform(queriesContext).toarray()
print(doc_tf_idf.shape)
cos_sim = cosine_similarity(query_vec, doc_tf_idf)
score = np.flip(cos_sim.argsort(), axis=1) # 排序文章再反序
print(score.shape)
#%% Rocchio

alpha = 1
beta = 0.8
rel_count = 5
iteration = 8
gemma = 0.15
non_rel_count = 1

for iter_num in range(iteration):
    print(iter_num)
    rel_vec = doc_tf_idf[score[:, :rel_count]].mean(axis=1) #欲新增至query的 TOP K doc vector 取mean
    non_rel_vec = doc_tf_idf[score[:, -non_rel_count:]].mean(axis=1) #最不相關的 K doc vector
    query_vec = alpha * query_vec + beta * rel_vec
    
    cos_sim = cosine_similarity(query_vec, doc_tf_idf)
    score = np.flip(cos_sim.argsort(axis=1), axis=1)
    
#%%
save_file_name = 'C:\\Users\\User\\Desktop\\NTUST\\IR\\result\\hw5-test.txt'
fp = open(save_file_name, "w")
fp.seek(0)
fp.write("Query,RetrievedDocuments\n")

for loop in range (len(queriesList)):
    write_string = queriesList[loop][0:-4] + ","
    for num , doc_id in enumerate(score[loop,:]):
        if num >= 5000:
            break
        write_string += docsList[doc_id][0:-4] + " "
    write_string += "\n"
    fp.write(write_string)
fp.truncate()
fp.close()
#%%