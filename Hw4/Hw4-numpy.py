#%%
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import numba as nb
import time
#%% 讀取檔案

queriesPath ='C:\\Users\\LAB\\Desktop\\2020-information-retrieval-and-applications-hw4-v2\\ntust-ir-2020_hw4_v2\\queries'
docsPath = 'C:\\Users\\LAB\\Desktop\\2020-information-retrieval-and-applications-hw4-v2\\ntust-ir-2020_hw4_v2\\docs'

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
#%% 計算各篇文章長度 ， 後面要用到
doc_len = np.zeros(len(docsContext))
for doc_id ,doc in enumerate(docsContext):
    words = doc.split()
    doc_len[doc_id] = len(words)
#%% tf matrix & get sparse matrix
vectorizer = CountVectorizer() #get tf-matrix
X = vectorizer.fit_transform(docsContext) #得到文章對於所有word的 tf - matrix
#%% dim_reduction
def dim_reduction (X , threshold):
    # 先找出tf-matrix 中 query的字 並且在做減維時不刪
    queryword = []
    for query in queriesContext:
        words = query.split()
        print(words)
        for word in words:
            if word not in queryword:
                queryword.append(word)
    docs , terms = X.shape
    X = X.toarray().astype(np.int16)
    print(docs)
    print(terms)
    # 找單字有大於threshold的值
    keep_term = X.sum(axis = 0) > threshold
    query_word_index = []
    print(np.sum(keep_term))
    #保留query的單字
    for word in queryword:
        query_index = vectorizer.get_feature_names().index(word)
        query_word_index.append(query_index)
        #print(str(query_index) + ',' + str(keep_term[query_index]))
        keep_term[query_index] = True
    word_term, = np.where(keep_term == True)
    print(np.sum(keep_term))
    # 拿到query在減為後的index位置
    query_index = {}
    for index in query_word_index:
        print(vectorizer.get_feature_names()[index])
        term_index = np.where(word_term == index)
        print(term_index[0][0])
        query_index[vectorizer.get_feature_names()[index]] = term_index[0][0]
    
    X = X[:, keep_term]
    return X ,query_index

reduce_array, query_index = dim_reduction(X , 100)
#%% PLSA
#np.sum(axis = 0) 對行做相加
#np.sum(axis = 1) 列相加
@nb.jit
def PLSA_numpy(tf_matrix , numOfTopic , numOfiter , docLength , vocLength ,doc_len):
    
    s1 = time.time()
    docSize = docLength
    vocSize = vocLength
    # Create the counter arrays.
    p_tk_dj = np.zeros([numOfTopic , docSize]) # P(tk | dj)
    p_wi_tk = np.zeros([vocSize , numOfTopic]) # P(wi | tk)
    p_tk_dj_wi = np.zeros([docSize, vocSize , numOfTopic]) # P(tk | dj, wi)
    # Initialize
    print ("Initializing...")
    # randomly assign values
    p_tk_dj = np.random.random(size = (numOfTopic , docSize))
    doc_toic_sum = np.sum(p_tk_dj,axis = 0)
    p_tk_dj = p_tk_dj / doc_toic_sum
    p_wi_tk = np.random.random(size = (vocSize , numOfTopic))
    term_topic_sum = np.sum(p_wi_tk , axis = 0)
    p_wi_tk = p_wi_tk / term_topic_sum
    for iteration in range(numOfiter) :
        iter_time = time.time()
        print("iter : "  + str(iteration))
        print('start E step')
        for doc in range(docSize) :
            this_doc_tk_dj = p_tk_dj[:,doc] #  得到 shape topic * 1
           # print(this_doc_tk_dj.shape)
            p_tk_widj_molecular = p_wi_tk * this_doc_tk_dj # p(wi|tk) * p(tk|dj)
           # print(p_tk_widj_molecular.shape)
            p_tk_widj_Denominator = np.sum(p_tk_widj_molecular, axis = 1) # sum_topic(p(wi|tk) * p(tk|dj))
            p_tk_widj_Denominator = np.reshape(p_tk_widj_Denominator,(vocSize,1))
           # print(p_tk_widj_Denominator.shape)
            this_doc_p_tk_widj = p_tk_widj_molecular / p_tk_widj_Denominator
           # print(this_doc_p_tk_widj.shape)
            p_tk_dj_wi[doc] = this_doc_p_tk_widj
            
        print('end E step')
        # m step
        print('start M step')
        # P(wi|tk)
        for topic in range(numOfTopic):
            print(topic)
            this_topic_wi_tk_molecular = tf_matrix * p_tk_dj_wi[:,:,topic] # 從p_tk_dj_wi取出 doc*voc的array， 與輸入的matrix做相乘 (doc * voc)
            this_topic_wi_tk_molecular = np.sum(this_topic_wi_tk_molecular , axis = 0) # 各個wi的算出的內容對所有doc為單位相加 (doc * voc ) 取行相加
            this_topic_wi_tk_Denominator = np.sum(this_topic_wi_tk_molecular)
            
            if this_topic_wi_tk_Denominator != 0:
                this_topic_wi_tk = this_topic_wi_tk_molecular / this_topic_wi_tk_Denominator
                this_topic_wi_tk = np.reshape(this_topic_wi_tk, (vocSize,))
            else:
                this_topic_wi_tk = np.zeros((vocSize,))
                
            p_wi_tk[:,topic] = this_topic_wi_tk
        print('end M step')
        # update p(tk|dj)
        print('update p(tk|dj)')
        for topic in range(numOfTopic):
            print(topic)
            this_topic_p_tk_dj_molecular = p_tk_dj_wi[:,:,topic] #P(tk|wi,dj)
            this_topic_p_tk_dj_molecular = this_topic_p_tk_dj_molecular * tf_matrix
            this_topic_p_tk_dj_molecular = np.sum(this_topic_p_tk_dj_molecular , axis = 1)
            this_topic_p_tk_dj_Denominator = doc_len
            this_topic_p_tk_dj = this_topic_p_tk_dj_molecular / this_topic_p_tk_dj_Denominator
            p_tk_dj[topic,:] = this_topic_p_tk_dj
        iter_end_time = time.time()
        print( 'iter use times : ' + str(iter_end_time - iter_time) + 's')
    final = time.time()
    print('total time : ' + str ((final - s1) / 60) + 'min')
    return p_tk_dj ,p_wi_tk
            
#%% run EM - algo
print(reduce_array.shape)
document_topic_prob, topic_word_prob = PLSA_numpy(reduce_array,24,40,reduce_array.shape[0],reduce_array.shape[1],doc_len)
#%%
queryword = []
for query in queriesContext:
    words = query.split()
    print(words)
    for word in words:
        if word not in queryword:
            queryword.append(word)
#%% cal_p(wi|dj)
p_wi_dj = np.zeros([14955,len(queryword)])
for i ,name in enumerate(queryword):
    index = query_index[name] # 得到query_word_index
    p_wi_dj[:,i] = reduce_array[:,index] / doc_len
#%% cal_ p(wi | BG)
p_wi_BG = np.zeros(len(queryword))
print(np.sum(reduce_array > 0))
totalword = np.sum(doc_len)
for index,doc in enumerate(queryword):
    tf = query_index[doc]
    p_wi_BG[index] = tf / totalword
#%% p(q|dj) 
p_q_dj = np.zeros([len(docsContext),len(queriesContext)])
alpha =0.38       #自訂的可調參數alpha
beta = 0.53

for num , query in enumerate(queriesContext):
    words = query.split() #這篇query幾個單字
    out_arr = np.zeros(len(docsContext))
    
    for index , word in enumerate (words):
        em = topic_word_prob[query_index[word],:].reshape(topic_word_prob.shape[1],-1) * document_topic_prob # sumation  p(wi|tk) * p(tk|dj) #all_topic 
        em = np.sum(em, axis = 0)
        middle = beta * em
        front = alpha * p_wi_dj[:, queryword.index(word)]
        back = (1 - alpha - beta) * p_wi_BG[queryword.index(word)]
        prob = front + middle + back
        if index == 0:
            out_arr = prob
        else:
            out_arr = out_arr * prob
            
    p_q_dj[: , num] = out_arr
#%%
res = {}
save_file_name = 'C:\\Users\\LAB\\Desktop\\2020-information-retrieval-and-applications-hw4-v2\\data\\test6.txt'
fp = open(save_file_name, "w")
fp.seek(0)
fp.write("Query,RetrievedDocuments\n")

for loop in range (len(queriesList)):
    write_string = queriesList[loop][0:-4] + ","
    for i,j in zip(docsList,p_q_dj[:,loop]):
        res[i[0:-4]] = j
    sorted_x = sorted(res.items(), key=lambda kv: kv[1],reverse = True)
    for iteration , doc in enumerate(sorted_x):
        if iteration >= 1000:
            break
        write_string += doc[0] + " "
    write_string += "\n"
    fp.write(write_string)
    res.clear()
    sorted_x.clear()
fp.truncate()
fp.close()
#%%
print(queryword.index('organ'))
        
