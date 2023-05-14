#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


match=pd.read_csv('matches.csv')
delivery=pd.read_csv('deliveries.csv.zip')


# In[3]:


delivery


# In[4]:


delivery.head()


# In[5]:


delivery.info()


# In[6]:


delivery.describe()


# In[7]:


delivery.isnull().sum()


# In[8]:


delivery.duplicated()


# In[9]:


match.shape


# In[10]:


delivery.shape


# In[11]:


total_score_df=delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
total_score_df


# In[12]:


total_score_df=total_score_df[total_score_df['inning']==1]


# In[13]:


total_score_df


# In[14]:


match_df=match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')
match_df


# In[15]:


match_df['team1'].unique()


# In[16]:


teams=[
    'Mumbai Indians',
    'channai super kings',
    'Sunrisers Hyderabad',
    'Royal Challengers Bangalore',
    'lakhnow super giant',
    'Delhi Capitals',
    'gujrat titans',
    'Rajasthan Royals',
    'Kings XI Punjab',
    'kolkata knightriders'
]


# In[17]:


match_df['team1']=match_df['team1'].str.replace('Delhi Daredevils','delhi capitals')
match_df['team2']=match_df['team2'].str.replace('Delhi Daredevils','delhi capitals')

match_df['team1']=match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2']=match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')


# In[18]:


match_df=match_df[match_df['team1'].isin(teams)]
match_df=match_df[match_df['team2'].isin(teams)]


# In[19]:


match_df.shape


# In[20]:


match_df=match_df[match_df['dl_applied']==0]
match_df


# In[21]:


delivery_df=match_df=match_df[['match_id','city','winner','total_runs']]


# In[22]:


delivery_df=match_df.merge(delivery,on='match_id')


# In[23]:


delivery_df=delivery_df[delivery_df['inning']==2]


# In[24]:


delivery_df.shape


# In[25]:


delivery_df['current_score']=delivery_df.groupby('match_id').cumsum()['total_runs_y']


# In[26]:


delivery_df['runs_left']=delivery_df['total_runs_x']+1-delivery_df['current_score']


# In[27]:


delivery_df['balls_left']=126-(delivery_df['over']*6+delivery_df['ball'])


# In[28]:


delivery_df


# In[ ]:





# In[29]:


delivery_df.tail()


# In[30]:


delivery_df['crr']=(delivery_df['current_score']*6)/(120-delivery_df['balls_left'])


# In[31]:


delivery_df['rrr']=(delivery_df['runs_left']*6)/delivery_df['balls_left']
delivery_df


# In[32]:


def result(row):
    return 1 if row['winner']==row['batting_team'] else 0


# In[33]:


delivery_df['result']=delivery_df.apply(result,axis=1)
delivery_df


# In[34]:


final_df=delivery_df[['batting_team','bowling_team','city','total_runs_x','runs_left','balls_left','crr','rrr','result']]


# In[35]:


final_df=final_df.sample(final_df.shape[0])


# In[36]:


final_df.sample()


# In[37]:


final_df.dropna(inplace=True)


# In[38]:


final_df = final_df[final_df['balls_left'] != 0]


# In[39]:


x=final_df.iloc[:,:-1]
y=final_df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# In[40]:


X_train


# In[41]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')


# In[42]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[43]:


pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])


# In[44]:


pipe.fit(X_train,y_train)


# In[45]:


y_pred = pipe.predict(x_test)


# In[46]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[47]:


pipe.predict_proba(x_test)[10]


# In[48]:


def match_summary(row):
    print("Batting Team-" + row['batting_team'] + " | Bowling Team-" + row['bowling_team'] + " | Target- " + str(row['total_runs_x']))


# In[49]:


def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team','bowling_team','city','runs_left','balls_left','total_runs_x','crr','rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
   
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','lose','win']]
    return temp_df,target


# In[60]:


temp_df,target = match_progression(delivery_df,54,pipe)
temp_df,target


# In[61]:


import matplotlib.pyplot as plt
plt.figure(figsize=(18,8))
# plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
plt.title('Target-' + str(target))


# In[57]:


teams


# In[58]:


delivery_df['city'].unique()


# In[ ]:




