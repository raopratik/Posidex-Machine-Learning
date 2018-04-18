
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# machine learning
from sklearn import svm  
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[2]:


data1 = pd.read_csv("final.csv")
data2 = pd.read_csv("final2.csv")


# In[3]:


data1['RIO_ACCEPT_REJECT']= data1['RIO_ACCEPT_REJECT'].map({'SM':1, 'NM': 0, 'M' : 0})
data2['RIO_ACCEPT_REJECT']= data2['RIO_ACCEPT_REJECT'].map({'SM':1, 'NM': 0, 'M' : 0})


# In[4]:


data1


# In[4]:


def counter(x):
    c=0
    y=x.split(',')
    #print(x)
    for i in y:
        if(i!=' '):
            c=c+1
    return c

def reduction(x):
    if x==-1:
        return 0
    else:
        return x

def cut(prob,i):
    return [1 if x >=i else 0 for x in prob]

def testresults(prob,testx,testy,model_var,name):
    Y_pred=cut(pd.DataFrame(model_var.predict_proba(testx))[1],prob)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(Y_pred,testy)]
    print((sum(correct)/len(correct))*100)
    tp = [1 if ((a == 1 and b == 1)) else 0 for (a, b) in zip(Y_pred,testy)]
    fp = [1 if ((a == 1 and b == 0)) else 0 for (a, b) in zip(Y_pred,testy)]
    fn= [1 if ((a == 0 and b == 1)) else 0 for (a, b) in zip(Y_pred,testy)]
    tn= [1 if ((a == 0 and b == 0)) else 0 for (a, b) in zip(Y_pred,testy)]
    tp=sum(tp)
    fp=sum(fp)
    fn=sum(fn)
    tn=sum(tn)
    print("Model:"+str(model_var))
    print(tp,fp,fn,tn)
    P=(tp)/(tp+fp)
    R=(tp)/(tp+fn)
    print("Precision:"+str(P*100),"\tRecall:"+str(R*100))
    F=2*P*R
    F=F/(P+R)
    print("F1 SCORE:"+str(F*100))
    
    syn1=pd.DataFrame({"MODEL":name,"PRECISION":P*100,"RECALL":R*100,"F1_SCORE":F*100,"POS":(tp+fn),"NEG":(fp+tn),"TP":tp,"FP":fp,"FN":fn,"TN":tn},index=[0])
    syn=syn1[['MODEL','POS','NEG','F1_SCORE','PRECISION','RECALL','TP','FP','FN','TN']]
    return syn

def adder(prob,testx,testy,model_var,syn,name):
    
    Y_pred=cut(pd.DataFrame(model_var.predict_proba(testx))[1],prob)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(Y_pred,testy)]
    print((sum(correct)/len(correct))*100)
    tp = [1 if ((a == 1 and b == 1)) else 0 for (a, b) in zip(Y_pred,testy)]
    fp = [1 if ((a == 1 and b == 0)) else 0 for (a, b) in zip(Y_pred,testy)]
    fn= [1 if ((a == 0 and b == 1)) else 0 for (a, b) in zip(Y_pred,testy)]
    tn= [1 if ((a == 0 and b == 0)) else 0 for (a, b) in zip(Y_pred,testy)]
    tp=sum(tp)
    fp=sum(fp)
    fn=sum(fn)
    tn=sum(tn)
    print("Model:"+str(model_var))
    print(tp,fp,fn,tn)
    P=(tp)/(tp+fp)
    R=(tp)/(tp+fn)
    print("Precision:"+str(P*100),"\tRecall:"+str(R*100))
    F=2*P*R
    F=F/(P+R)
    print("F1 SCORE:"+str(F*100))
    df2 = pd.DataFrame({"MODEL":name,"PRECISION":P*100,"RECALL":R*100,"F1_SCORE":F*100,"POS":(tp+fn),"NEG":(fp+tn),"TP":tp,"FP":fp,"FN":fn,"TN":tn},index=[len(syn)])
    df3=df2[['MODEL','POS','NEG','F1_SCORE','PRECISION','RECALL','TP','FP','FN','TN']]
    syn=syn.append(df3)
    print(len(syn))
    return syn

def adder2(prob,testx,testy,model_var,syn,name):
    
    Y_pred=model_var.predict(testx)
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(Y_pred,testy)]
    print((sum(correct)/len(correct))*100)
    tp = [1 if ((a == 1 and b == 1)) else 0 for (a, b) in zip(Y_pred,testy)]
    fp = [1 if ((a == 1 and b == 0)) else 0 for (a, b) in zip(Y_pred,testy)]
    fn= [1 if ((a == 0 and b == 1)) else 0 for (a, b) in zip(Y_pred,testy)]
    tn= [1 if ((a == 0 and b == 0)) else 0 for (a, b) in zip(Y_pred,testy)]
    tp=sum(tp)
    fp=sum(fp)
    fn=sum(fn)
    tn=sum(tn)
    print("Model:"+str(model_var))
    print(tp,fp,fn,tn)
    P=(tp)/(tp+fp)
    R=(tp)/(tp+fn)
    print("Precision:"+str(P*100),"\tRecall:"+str(R*100))
    F=2*P*R
    F=F/(P+R)
    print("F1 SCORE:"+str(F*100))
    df2 = pd.DataFrame({"MODEL":name,"PRECISION":P*100,"RECALL":R*100,"F1_SCORE":F*100,"POS":(tp+fn),"NEG":(fp+tn),"TP":tp,"FP":fp,"FN":fn,"TN":tn},index=[len(syn)])
    df3=df2[['MODEL','POS','NEG','F1_SCORE','PRECISION','RECALL','TP','FP','FN','TN']]
    syn=syn.append(df3)
    print(len(syn))
    return syn


# In[5]:


data1['MATCH_TYPE2']=data1['MATCH_TYPE'].apply(counter)
data2['MATCH_TYPE2']=data2['MATCH_TYPE'].apply(counter)
data1['COL1040STRENGTH']=data1['COL1040STRENGTH'].apply(reduction)
data2['COL1040STRENGTH']=data2['COL1040STRENGTH'].apply(reduction)
data1['COL1050STRENGTH']=data1['COL1050STRENGTH'].apply(reduction)
data2['COL1050STRENGTH']=data2['COL1050STRENGTH'].apply(reduction)
data1['COL2000STRENGTH']=data1['COL2000STRENGTH'].apply(reduction)
data2['COL2000STRENGTH']=data2['COL2000STRENGTH'].apply(reduction)
data1['COL1000LVL']=data1['COL1000LVL'].apply(reduction)
data2['COL1000LVL']=data2['COL1000LVL'].apply(reduction)


# In[6]:


data1['ADDRMINLENSTRENGTH']=(data1['ADDR1MINLENSTRENGTH']+data1['ADDR2MINLENSTRENGTH']+data1['ADDR3MINLENSTRENGTH'])/3
data2['ADDRMINLENSTRENGTH']=(data2['ADDR1MINLENSTRENGTH']+data2['ADDR2MINLENSTRENGTH']+data2['ADDR3MINLENSTRENGTH'])/3
data1['ADDRMAXLENSTRENGTH']=(data1['ADDR1MAXLENSTRENGTH']+data1['ADDR2MAXLENSTRENGTH']+data1['ADDR3MAXLENSTRENGTH'])/3
data2['ADDRMAXLENSTRENGTH']=(data2['ADDR1MAXLENSTRENGTH']+data2['ADDR2MAXLENSTRENGTH']+data2['ADDR3MAXLENSTRENGTH'])/3
data1['ADDRESSMATCHSTRENGTH']=(data1['ADDRESSMATCH1STRENGTH']+data1['ADDRESSMATCH2STRENGTH']+data1['ADDRESSMATCH3STRENGTH'])/3
data2['ADDRESSMATCHSTRENGTH']=(data2['ADDRESSMATCH1STRENGTH']+data2['ADDRESSMATCH2STRENGTH']+data2['ADDRESSMATCH3STRENGTH'])/3

data1['DS']=data1['COL4000STRENGTH']**2
data1['MS']=(data1['COL2000STRENGTH']**2)/10
data2['DS']=data2['COL4000STRENGTH']**2
data2['MS']=(data2['COL2000STRENGTH']**2)/10


# In[7]:


data1=data1[['MS','DS','COL1050STRENGTH','HNO1STRENGTH','ADDRMINLENSTRENGTH','ADDRMAXLENSTRENGTH','ADDR2MAXLENSTRENGTH','ADDR3MAXLENSTRENGTH','SCALE_TYPE','ADDRESSMATCHSTRENGTH','COL5000STRENGTH','COL4000STRENGTH','MATCH_TYPE2','COL2000STRENGTH','COL5040STRENGTH','COL1000STRENGTH'
,'CS1STRENGTH','CS2STRENGTH','COL1000LVL','COL3000STRENGTH','COL5030STRENGTH','COL1040STRENGTH','CS3STRENGTH','RIO_ACCEPT_REJECT']]

data2=data2[['MS','DS','COL1050STRENGTH','HNO1STRENGTH','ADDRMINLENSTRENGTH','ADDRMAXLENSTRENGTH','ADDR2MAXLENSTRENGTH','ADDR3MAXLENSTRENGTH','SCALE_TYPE','ADDRESSMATCHSTRENGTH','COL5000STRENGTH','COL4000STRENGTH','MATCH_TYPE2','COL2000STRENGTH','COL5040STRENGTH','COL1000STRENGTH'
,'CS1STRENGTH','CS2STRENGTH','COL1000LVL','COL3000STRENGTH','COL5030STRENGTH','COL1040STRENGTH','CS3STRENGTH','RIO_ACCEPT_REJECT']]

#data1=data1.dropna()
#data2=data2.dropna()
frames = [data2,data1]

result = pd.concat(frames)
len(result),len(result.dropna())
#train, test = train_test_split(train, test_size = 0.2)
#train=data1
#test=data2


# In[8]:


#print(result[['SRCCOL11','MATCH_TYPE2']])
from sklearn.model_selection import train_test_split

train, test = train_test_split(result, test_size = 0.2)


# In[9]:


train=train.dropna()
print(len(train))

#imp1=train[train['RIO_ACCEPT_REJECT']==1]
#frames=[imp1,train]
#train=pd.concat(frames)
trainx=train[['COL1050STRENGTH','COL5000STRENGTH','ADDRMAXLENSTRENGTH','SCALE_TYPE','ADDRMINLENSTRENGTH','ADDRESSMATCHSTRENGTH','COL4000STRENGTH','MATCH_TYPE2','COL2000STRENGTH','COL5040STRENGTH','COL1000STRENGTH'
,'CS1STRENGTH','CS2STRENGTH','COL1000LVL','COL3000STRENGTH','COL5030STRENGTH','COL1040STRENGTH','CS3STRENGTH']]
trainy=train['RIO_ACCEPT_REJECT']

trainx.shape


# In[10]:


from imblearn.over_sampling import SMOTE

sm=SMOTE(ratio='auto',kind="regular")
X_res, y_res = sm.fit_sample(trainx,trainy)
x=pd.DataFrame(X_res)
y=pd.DataFrame(y_res)
print(len(y[y[0]==0]))
sm1=x[y[0]==1]
sm1[['COL1050STRENGTH','COL5000STRENGTH','ADDRMAXLENSTRENGTH','SCALE_TYPE','ADDRMINLENSTRENGTH','ADDRESSMATCHSTRENGTH','COL4000STRENGTH','MATCH_TYPE2','COL2000STRENGTH','COL5040STRENGTH','COL1000STRENGTH'
,'CS1STRENGTH','CS2STRENGTH','COL1000LVL','COL3000STRENGTH','COL5030STRENGTH','COL1040STRENGTH','CS3STRENGTH']]=x[y[0]==1]
smfin=sm1[['COL1050STRENGTH','COL5000STRENGTH','ADDRMAXLENSTRENGTH','SCALE_TYPE','ADDRMINLENSTRENGTH','ADDRESSMATCHSTRENGTH','COL4000STRENGTH','MATCH_TYPE2','COL2000STRENGTH','COL5040STRENGTH','COL1000STRENGTH'
,'CS1STRENGTH','CS2STRENGTH','COL1000LVL','COL3000STRENGTH','COL5030STRENGTH','COL1040STRENGTH','CS3STRENGTH']]

smy=y[y[0]==1]
smy['RIO_ACCEPT_REJECT']=smy
smyfin=smy['RIO_ACCEPT_REJECT']

framesx=[trainx,smfin[:20000]]
framesy=[trainy,smyfin[:20000]]
trainx=pd.concat(framesx)
trainy=pd.concat(framesy)


# In[11]:


test=test.dropna()
testx=test[['COL1050STRENGTH','COL5000STRENGTH','ADDRMAXLENSTRENGTH','SCALE_TYPE','ADDRMINLENSTRENGTH','ADDRESSMATCHSTRENGTH','COL4000STRENGTH','MATCH_TYPE2','COL2000STRENGTH','COL5040STRENGTH','COL1000STRENGTH','CS1STRENGTH','CS2STRENGTH','COL1000LVL','COL3000STRENGTH','COL5030STRENGTH','COL1040STRENGTH','CS3STRENGTH']]
testy=test['RIO_ACCEPT_REJECT']
testx.shape


# In[12]:


#LOGISTIC
logreg = LogisticRegression()
logreg.fit(trainx,trainy)

acc_log = round(logreg.score(trainx,trainy) * 100, 2)
acc_log


# In[13]:


get_ipython().run_cell_magic('time', '', 'Y_pred = logreg.predict(testx)')


# In[14]:


syn=testresults(.5,testx,testy,logreg,"LOGISTIC_REGRESSION")


# In[15]:


svc = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)  
        
svc.fit(trainx, trainy) 
#svc = SVC()
#svc.fit(trainx, trainy)

acc_svc = round(svc.score(trainx, trainy) * 100, 2)
acc_svc


# In[16]:


get_ipython().run_cell_magic('time', '', 'Y_pred = svc.predict(testx)')


# In[17]:


syn=adder2(.5,testx,testy,svc,syn,'SUPPORT_VECTOR_MACHINES')


# In[18]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(trainx, trainy)

acc_knn = round(knn.score(trainx, trainy) * 100, 2)
acc_knn


# In[19]:


get_ipython().run_cell_magic('time', '', 'Y_pred = knn.predict(testx)')


# In[20]:


syn=adder(.5,testx,testy,knn,syn,'K_NEIGHBOURS')


# In[21]:


gaussian = GaussianNB()
gaussian.fit(trainx, trainy)

acc_gaussian = round(gaussian.score(trainx, trainy) * 100, 2)
acc_gaussian


# In[22]:


get_ipython().run_cell_magic('time', '', 'Y_pred = gaussian.predict(testx)')


# In[23]:


syn=adder(.5,testx,testy,gaussian,syn,'GAUSSIAN')


# In[24]:


perceptron = Perceptron()
perceptron.fit(trainx, trainy)

acc_perceptron = round(perceptron.score(trainx, trainy) * 100, 2)
acc_perceptron


# In[25]:


get_ipython().run_cell_magic('time', '', 'Y_pred = perceptron.predict(testx)')


# In[26]:



syn=adder2(.5,testx,testy,perceptron,syn,"PERCEPTRON")


# In[27]:


sgd = SGDClassifier()
sgd.fit(trainx, trainy)

acc_sgd = round(sgd.score(trainx, trainy) * 100, 2)


# In[28]:


get_ipython().run_cell_magic('time', '', 'Y_pred = sgd.predict(testx)')


# In[29]:


syn=adder2(.5,testx,testy,sgd,syn,"SGD")


# In[30]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(trainx, trainy)

acc_decision_tree = round(decision_tree.score(trainx, trainy) * 100, 2)
acc_decision_tree


# In[31]:


get_ipython().run_cell_magic('time', '', 'Y_pred = decision_tree.predict(testx)')


# In[32]:


syn=adder(.5,testx,testy,decision_tree,syn,'DECISION_TREE')


# In[33]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(trainx, trainy)

random_forest.score(trainx, trainy)
acc_random_forest = round(random_forest.score(trainx, trainy) * 100, 2)
acc_random_forest


# In[34]:


get_ipython().run_cell_magic('time', '', 'Y_pred = random_forest.predict(testx)')


# In[35]:


random_forest
syn=adder(.5,testx,testy,random_forest,syn,'RANDOM_FOREST')


# In[36]:


print(syn)


# In[37]:


logreg.coef_


# In[38]:


r=knn.kneighbors(X=testx, n_neighbors=3, return_distance=True)


# In[39]:


trainx.shape


# In[40]:


r[0],r[1]


# In[41]:


(testy==1).sum()


# In[42]:


(testy==0).sum()


# In[43]:


len(testy)


# In[44]:


len(trainy)


# In[ ]:




