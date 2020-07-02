#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[2]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


titanic=pd.read_csv('train.csv')


# In[ ]:


titanic.head()


# In[ ]:


col=titanic.columns
print (col)


# In[ ]:


titanic.shape


# In[ ]:


titanic.info


# In[ ]:


titanic.describe()


# In[ ]:


[print(columns,np.unique(titanic[columns].isnull())) for columns in col]


# In[ ]:


list((columns,np.unique(titanic[columns].astype(str)).size) for columns in col)


# In[ ]:


print ("total_females =",len(titanic[(titanic["Sex"]=="female")].index))
print ("total_males =",len(titanic[(titanic["Sex"]=="male")].index))


# In[ ]:


survived=titanic[(titanic["Sex"]=="male") & (titanic["Survived"]==1)]
len(survived.index)


# In[ ]:


survived=titanic[(titanic["Sex"]=="female") & (titanic["Survived"]==1)]
len(survived.index)


# In[ ]:


print([titanic.groupby("Sex")["Survived"].value_counts(normalize = True)])


# In[ ]:


class_pivot=titanic.pivot_table(index="Sex",values="Survived")
class_pivot.plot.bar()
plt.show()


# In[ ]:


print("Pclass_1=",len(titanic[(titanic["Pclass"]==1)].index))
print("Pclass_2=",len(titanic[(titanic["Pclass"]==2)].index))
print("Pclass_3=",len(titanic[(titanic["Pclass"]==3)].index))


# In[ ]:


tot_pass=titanic["Pclass"].value_counts().sort_index()
print(tot_pass)


# In[ ]:


survived_class =[titanic.groupby("Pclass")["Survived"].sum()]
print(survived_class)


# In[ ]:


print([titanic.groupby("Pclass")["Survived"].value_counts(normalize=True)])


# In[ ]:


class_pivot=titanic.pivot_table(index="Pclass",values="Survived")
class_pivot.plot.bar()
plt.show()


# In[ ]:


coorelation_matrix=titanic.corr(method='pearson')
coorelation_matrix


# In[ ]:


survived=titanic[titanic["Survived"]==1]
died= titanic[titanic["Survived"]==0]
survived["Age"].plot.hist(color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()


# In[ ]:


titanic_copy=titanic
titanic_copy["Age"]=titanic_copy["Age"].fillna(titanic['Age'].mean(),inplace=False)


# In[ ]:


titanic_copy.shape


# In[ ]:


class_pivot=titanic.pivot_table(index="Embarked",values="Survived")
class_pivot.plot.bar()
plt.show()


# In[ ]:


titanic_copy=titanic_copy.dropna(subset=["Sex","Pclass","Embarked","Age"])


# In[ ]:


titanic_copy.shape


# In[ ]:


Y_titanic=titanic_copy.loc[:,"Survived"]
X_titanic=titanic_copy.loc[:,["Age","Sex","Pclass","Embarked"]]


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
age = scaler.fit_transform(X_titanic["Age"].values.reshape(-1,1))
X_titanic["Age"]=age


# In[ ]:


X_titanic_he=pd.get_dummies(X_titanic,columns=["Pclass","Sex","Embarked"])
X_titanic_he.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_titanic_he,Y_titanic,test_size=0.3,random_state=0)


# In[ ]:


from sklearn import tree
clf=tree.DecisionTreeClassifier(min_samples_split=70,min_samples_leaf=10)
clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_pred)


# In[ ]:


from sklearn.metrics import precision_score
precision_score(Y_test,Y_pred)


# In[ ]:


from sklearn.metrics import recall_score
recall_score(Y_test,Y_pred)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100,min_samples_split=50)
clf=clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)

accuracy_score(Y_test,Y_pred)


# In[ ]:


print("Precision=",precision_score(Y_test,Y_pred))
print("Recall=",recall_score(Y_test,Y_pred))


# In[ ]:


from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)

accuracy_score(Y_test,Y_pred)


# In[ ]:


print("Precision=",precision_score(Y_test,Y_pred))
print("Recall=",recall_score(Y_test,Y_pred))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)

accuracy_score(Y_test,Y_pred)


# In[ ]:


print("Precision=",precision_score(Y_test,Y_pred))
print("Recall=",recall_score(Y_test,Y_pred))


# In[ ]:


from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0,solver='sag',multi_class='multinomial').fit(X_train,Y_train)
Y_pred=clf.predict(X_test)
accuracy_score(Y_test,Y_pred)


# In[ ]:


print("Precision=",precision_score(Y_test,Y_pred))
print("Recall=",recall_score(Y_test,Y_pred))


# In[ ]:


from sklearn import svm
clf=svm.SVC().fit(X_train,Y_train)
Y_pred=clf.predict(X_test)

accuracy_score(Y_test,Y_pred)


# In[ ]:


print("Precision=",precision_score(Y_test,Y_pred))
print("Recall=",recall_score(Y_test,Y_pred))


# In[ ]:




