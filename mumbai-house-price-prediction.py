#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)


# In[2]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


df1=pd.read_csv("/kaggle/input/mumbai-house-prices/Mumbai House Prices.csv")


# In[4]:


df1.head()


# In[5]:


df1.shape


# In[6]:


df1.columns


# In[7]:


df1['locality'].unique()


# In[8]:


df1['region'].unique()


# In[9]:


df1['type'].unique()


# In[10]:


df2=df1.drop(['locality','status'],axis='columns')
df2.shape


# In[11]:


df2.head()


# In[12]:


df2["Amount"] = df2['price'].astype(str) +" "+ df2["price_unit"]
df2.head()


# In[13]:


def fun(x):
    if 'Cr' in x or 'cr' in x:
        s=str(x).split(" ")[0]
        s1=str(int(float(s)*100))
        return s1
    else:
        s=str(x).split(" ")[0]
        return s


# In[14]:


df2['Amount']=df2['Amount'].map(lambda x:fun(x))


# In[15]:


df2['Amount']


# In[16]:


df2.head()


# In[17]:


df3=df2.drop(['price','price_unit'],axis='columns')
df3.shape


# In[18]:


df3.head()


# In[19]:


df4 = df3.rename({'Amount': 'Price_in_Lakhs'}, axis='columns')
df4.head()


# In[20]:


df4.isnull().sum()


# In[21]:


df4['bhk'].unique()


# In[22]:


df4.dtypes


# In[23]:


df4['Price_in_Lakhs'] = df4['Price_in_Lakhs'].astype('float')
df4.dtypes


# In[24]:


df5=df4.copy()
df5['Price_per_sqft']=df5['Price_in_Lakhs']*100000/df5['area']
df5.head()


# In[25]:


df6=df5.replace('Unknown',value=np.NaN)
df6.head()


# In[26]:


df6.isnull().sum()


# In[27]:


df7=df6.dropna()
df7.shape


# In[28]:


df7.isnull().sum()


# In[29]:


len(df7.region.unique())


# In[30]:


df7.region


# In[31]:


df7.region = df7.region.apply(lambda x: x.strip())
region_stats = df7['region'].value_counts(ascending=False)
region_stats


# In[32]:


len(region_stats[region_stats>10])


# In[33]:


len(region_stats[region_stats<=10])


# In[34]:


region_stats_less_than_10 = region_stats[region_stats<=10]
region_stats_less_than_10


# In[35]:


df7.region = df7.region.apply(lambda x: 'other' if x in region_stats_less_than_10 else x)
len(df7.region.unique())


# In[36]:


df7.head()


# In[37]:


df7.dtypes


# In[38]:


df7[df7.area/df7.bhk<250].head()


# In[39]:


df8 = df7[~(df7.area/df7.bhk<250)]
df8.shape


# In[40]:


df8.Price_per_sqft.describe()


# In[41]:


df8.shape


# In[42]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('region'):
        m = np.mean(subdf.Price_per_sqft)
        st = np.std(subdf.Price_per_sqft)
        reduced_df = subdf[(subdf.Price_per_sqft>(m-st)) & (subdf.Price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df9 = remove_pps_outliers(df8)
df9.shape


# In[43]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df9.Price_per_sqft,rwidth=0.8)
plt.xlabel("Price per sqft")
plt.ylabel("Count")


# In[44]:


df9['region'].unique()


# In[45]:


df9.head(10)


# In[46]:


df10=df9.copy()
df10.head()


# In[47]:


import matplotlib.pyplot as plt
import matplotlib

def chart(df, region):
    bhk2 = df[(df.region == region) & (df.bhk == 2)]
    bhk3 = df[(df.region == region) & (df.bhk == 3)]
    
    matplotlib.rcParams['figure.figsize'] = (15, 10)
    
    plt.scatter(bhk2.area, bhk2.Price_in_Lakhs, color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.area, bhk3.Price_in_Lakhs, marker='+', color='green', label='3 BHK', s=50)
    
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(region)
    plt.legend()

# Assuming you have defined the 'df10' DataFrame earlier
chart(df10, "Agripada")
plt.show()


# In[48]:


chart(df10,'Mira Road East')


# In[49]:


df10.head()
df10.shape


# In[50]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for region, region_df in df.groupby('region'):
        global bhk_stats
        bhk_stats = {}
        for bhk, bhk_df in region_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.Price_per_sqft),
                'std': np.std(bhk_df.Price_per_sqft),
                'count': bhk_df.shape[0]
                
            }
        for bhk, bhk_df in region_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.Price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df11 = remove_bhk_outliers(df10)
print(df11.shape)


# In[51]:


bhk_stats


# In[52]:


chart(df11,'Mira Road East')


# In[53]:


chart(df11,'Agripada')


# In[54]:


df11.head()


# In[55]:


plt.scatter(df11.age,df11.Price_in_Lakhs)


# In[56]:


df12=df11.drop(['Price_per_sqft'],axis='columns')
df12.head()


# In[57]:


dummies=pd.get_dummies(df12.region)
dummies.head(20)


# In[58]:


df13=pd.concat([df12,dummies.drop('other',axis='columns')],axis='columns')
df13.head()


# In[59]:


df14=df13.drop('region',axis='columns')
df14.head()


# In[60]:


x=df14.drop(['Price_in_Lakhs'],axis='columns')
x.head(10)


# In[61]:


dummies_2=pd.get_dummies(df14.type)
dummies_2.head(10)


# In[62]:


df15=pd.concat([df14,dummies_2.drop('Studio Apartment',axis='columns')],axis='columns')
df15.head()


# In[63]:


df16=df15.drop('type',axis='columns')
df16.head()


# In[64]:


dummies_3=pd.get_dummies(df16.age)
dummies_3.head(10)


# In[65]:


df17=pd.concat([df16,dummies_3.drop('Resale',axis='columns')],axis='columns')
df17.head()


# In[66]:


df18=df17.drop('age',axis='columns')
df18.head()


# In[67]:


df18.shape


# In[68]:


x=df18.drop(['Price_in_Lakhs'],axis='columns')
x.head()


# In[69]:


x.shape


# In[70]:


y=df18.Price_in_Lakhs
y.head()


# In[71]:


len(y)


# In[72]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)


# In[73]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(x_train,y_train)
lr_clf.score(x_test,y_test)


# In[74]:


from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.linear_model import LinearRegression

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
model = LinearRegression()

scores = cross_val_score(model, x, y, cv=cv)

print("Cross-validation scores:", scores)
print("Mean score:", scores.mean())
print("Standard deviation:", scores.std())


# In[75]:


x.columns


# In[76]:


df18.head()


# In[77]:


location='Agripada'
np.where(x.columns==location)[0][0]


# In[78]:


def predict_price(location,sqft,bhk):
    loc_index = np.where(x.columns==location)[0][0]
    
    X=np.zeros(len(x.columns))
    X[0] = bhk
    X[1] = sqft
    if loc_index >= 0:
        X[loc_index] = 1
        
    return lr_clf.predict([X])[0]
    


# In[79]:


predict_price('Agripada',600, 2)


# In[80]:


predict_price('Andheri West',600, 2)


# In[81]:


predict_price('Airoli',1233, 4)


# In[82]:


import pickle
with open('Mumbai_house_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# In[83]:


import json
columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))

