#%%
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import numba as nb
import time
import cupy as cp
#%% 讀取檔案

queriesPath ='C:\\Users\\User\\Desktop\\NTUST\\IR\\data\\ntust-ir-2020_hw4_v2\\queries'
docsPath = 'C:\\Users\\User\\Desktop\\NTUST\\IR\\data\\ntust-ir-2020_hw4_v2\\docs'

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
vectorizer = CountVectorizer(stop_words=None, token_pattern="(?u)\\b\\w+\\b") #get tf-matrix
X = vectorizer.fit_transform(docsContext) #得到文章對於所有word的 tf - matrix
print(X.shape)
#%%
#@nb.jit()
# axis = 0 保留行
# axis = 1 保留列
def PLSA_sparseMatrix(matrix , iteration , numOfTopic):
    sparse = coo_matrix(matrix)
    sparse_data = sparse.data
    sparse_row = sparse.row
    sparse_col = sparse.col
    sparse_size = sparse_data.shape[0]
    p_tk_wi_dj = cp.zeros((sparse_size , numOfTopic))
    p_tk_dj = cp.zeros((numOfTopic , matrix.shape[0]))
    p_wi_tk = cp.zeros(( matrix.shape[1] , numOfTopic))
    print ("Initializing...")
    # randomly assign values
    p_tk_dj = cp.random.random(size = (numOfTopic , matrix.shape[0]))
    doc_toic_sum = cp.sum(p_tk_dj,axis = 0)
    p_tk_dj = p_tk_dj / doc_toic_sum
    p_wi_tk = cp.random.random(size = (matrix.shape[1] , numOfTopic))
    term_topic_sum = cp.sum(p_wi_tk , axis = 0)
    p_wi_tk = p_wi_tk / term_topic_sum
    
    for loop in range(iteration):
        # E_step
        print('iter : '  + str(loop))
        s1 = time.time()
        print('E step')
        for index in range(len(sparse_data)):
            p_tk_wi_dj[index,:] = p_wi_tk[sparse_col[index] , :] * p_tk_dj[ : , sparse_row[index]]
        topic_normal = cp.sum(p_tk_wi_dj , axis = 1)
        for topic in range(numOfTopic):
            p_tk_wi_dj [:,topic] = p_tk_wi_dj [:,topic] / topic_normal
        print('end E step')
        # M step
        print('M step')
        M_step_molecular = sparse_data[:,np.newaxis] * p_tk_wi_dj
        for index in range(len(sparse_data)):
            #if index % 10000 == 0:
                #print(index)
            p_wi_tk[sparse_col[index],:] += M_step_molecular[index,:]
        m_step_normalize = cp.sum(p_wi_tk, axis = 0)[np.newaxis,:]
        print(cp.sum(p_wi_tk, axis = 0)[cp.newaxis,:].shape)
        p_wi_tk /= m_step_normalize
        print('end M step')
        # update p(tk|dj)
        print('update p(tk|dj)')
        for index in range(len(sparse_data)):
            doc_index = sparse_row[index]
            p_tk_dj[:,doc_index] += M_step_molecular[index,:]
        tk_dj_normalize = cp.sum(p_tk_dj, axis = 0)[cp.newaxis,:]
        print(np.sum(p_tk_dj, axis = 0)[cp.newaxis,:].shape)
        p_tk_dj /= tk_dj_normalize
        print('end p(tk|dj)')
        s2 = time.time()
        print(str(loop)  + ' time : ' , str(s2 - s1) + 's')
        
    return p_tk_dj,p_wi_tk
#%%
p_tk_dj,p_wi_tk = PLSA_sparseMatrix(X,45,128)
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
print(p_wi_dj[:,0].shape)
for i ,name in enumerate(queryword):
    index = vectorizer.get_feature_names().index(name) # 得到query_word_index
    p_wi_dj[:,i] = X[:,index].toarray().reshape(14955)
    p_wi_dj[:,i] /= doc_len
#%% cal_ p(wi | BG)
p_wi_BG = np.zeros(len(queryword))
totalword = np.sum(doc_len)
for i,name in enumerate(queryword):
    index = vectorizer.get_feature_names().index(name)
    p_wi_BG[i] = np.sum(X[:,index].toarray()) / totalword
#%% p(q|dj) 
p_q_dj = np.zeros([len(docsContext),len(queriesContext)])
alpha =0.7       #自訂的可調參數alpha
beta = 0.2

for num , query in enumerate(queriesContext):
    words = query.split() #這篇query幾個單字
    out_arr = np.zeros(len(docsContext))
    
    for index , word in enumerate (words):
        w_index = vectorizer.get_feature_names().index(word)
        em = p_wi_tk[w_index,:].reshape(p_wi_tk.shape[1],-1) * p_tk_dj # sumation  p(wi|tk) * p(tk|dj) #all_topic 
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
save_file_name = 'C:\\Users\\ipprlab\\Desktop\\2020-information-retrieval-and-applications-hw4-v2\\sparse_test2.txt'
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