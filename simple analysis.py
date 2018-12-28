
# coding: utf-8

# ## Comparing Results
# 
# I am compring differnt prediction models based on mean of absoulote error (MAE) and mean of square error over (MSE) the test set. The best model is random forest.  

# In[62]:


compare_result(y_pred,y_test)


# ### Libraries

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression,Lasso
from sklearn import neighbors
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
get_ipython().magic('matplotlib inline')


# In[2]:


# read Date
director =  '/Users/z002df6/Downloads/'
fln = 'data.xlsx'
df = pd.read_excel(director+fln)
df.head()


# In[26]:


inputs = ['f','d']+list(map(lambda x: 'i'+str(x),range(1,19)))

fig, axes = plt.subplots(nrows=20, ncols=7, figsize=(30, 20))
for i in range(1,8):
    for j,name in enumerate(inputs):
        ax = df.plot.scatter( name,'o'+str(i),
                             ax = axes[j][i-1])
        ax.set(xlabel= name, ylabel='o'+str(i))
plt.tight_layout()
plt.show()


# In[3]:


def CorrMtx(df, dropDuplicates = True):

    df = df.corr()

    # Exclude duplicate correlations by masking uper right values
    if dropDuplicates:    
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    # Set background color / chart style
    sns.set_style(style = 'white')

    # Set up  matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Add diverging colormap from red to blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Draw correlation plot with or without duplicates
    if dropDuplicates:
        sns.heatmap(df, mask=mask, cmap=cmap, 
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
    else:
        sns.heatmap(df, cmap=cmap, 
                square=True,
                linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)


CorrMtx(df, dropDuplicates = True)


# In[4]:


inputs  = ['f','d']+list(map(lambda x: 'i'+str(x),range(1,19)))
outputs = ['o'+str(i) for i in range(1,8)]
X = np.array(df[inputs])
y = np.array(df[outputs])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
y_pred = {}


# In[5]:


df[outputs].describe()


# ### Crude Average and Linear Model

# In[39]:


avr = y_train.mean(axis=0)
avr = avr[np.newaxis,:]
y_pred['Crude Avrage'] = np.repeat(avr, y_test.shape[0], axis=0)
LM = Lasso(alpha=0.1)
LM.fit(X_train, y_train)
y_pred['Lasso'] = LM.predict(X_test)


# ### Random Forest and K-Nearest Neighbour

# In[41]:


rf = RandomForestRegressor(n_estimators = 100, max_depth = 10)
# Train the model on training data
rf.fit(X_train, y_train)
y_pred['Random Forest'] = rf.predict(X_test)


# In[42]:


knn = neighbors.KNeighborsRegressor(70, weights='distance')
knn.fit(X_train, y_train)
y_pred['knn'] = knn.predict(X_test)


# ### Deep Neural Net

# In[15]:


DNN_model = Sequential()
DNN_model.add(Dense(100,input_dim=X_train.shape[1],init='uniform',activation='relu'))
DNN_model.add(Dropout(0.5))
DNN_model.add(Dense(50,init='uniform',activation='tanh'))
DNN_model.add(Dropout(0.5))
DNN_model.add(Dense(100,init='uniform',activation='relu'))
DNN_model.add(Dropout(0.5))
DNN_model.add(Dense(7,init='uniform',activation='relu'))
DNN_model.add(Dropout(0.5))
DNN_model.add(Dense(7,init='uniform',activation='linear'))
DNN_model.summary()


# ### Fitting the DNN

mn = X_train.mean(axis=0)
sd = X_train.mean(axis=0)
#model.compile(loss='mean_absolute_error',optimizer='adam',metrics='[accuracy]')
DNN_model.compile(loss='mean_squared_error',optimizer='adam')
history = DNN_model.fit((X_train-mn)/sd,y_train,  
                    validation_data=((X_test-mn)/sd, y_test),
                    epochs =500,
                    batch_size=100,
                    verbose=2)
y_pred['DNN'] = DNN_model.predict((X_test-mn)/sd)



# In[43]:


plt.figure(figsize=(10, 8))
plt.title("Dense model training", fontsize=12)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Test")
plt.grid("on")
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("loss", fontsize=12)
plt.legend(loc="upper right")


# In[20]:


class Result():
    def __init__(self):
        self.result = pd.DataFrame({'model': [],'MAE' :  [],'MSE': []})
    def update(self, model_name,y_test,y_pred ):
        new_df =pd.DataFrame({'MSE': round(mean_absolute_error(y_pred,y_test),2) ,
                                'MAE' :  round(sqrt(mean_squared_error(y_pred,y_test)),2),
                                'model': model_name},index = [1])
        self.result = self.result.append(new_df, ignore_index=True)
    def get(self):
        return self.result.set_index('model').sort_values('MAE',ascending =False)
    


# In[55]:


def compare_result(y_pred,y_test):
    '''
        compare prediction for test set in different models

    '''
    result = pd.DataFrame([{'Model': k ,
                            'MAE': round(mean_absolute_error(y_pred[k],y_test),2),
                            'RMSE' :  round(sqrt(mean_squared_error(y_pred[k],y_test)),2)} 
                            for k in y_pred.keys()
                          ])
    return result[['Model','MAE','RMSE']].set_index('Model').sort_values('MAE',ascending =False)    


# In[56]:


y_pred = {}
y_pred['Lasso'] = LM.predict(X_test)
y_pred['Crude Average'] = np.repeat(avr, y_test.shape[0], axis=0)
y_pred['Random Forest'] = rf.predict(X_test)
y_pred['DNN'] = DNN_model.predict((X_test-mn)/sd)
y_pred['KNN'] = knn.predict(X_test)
compare_result(y_pred,y_test)

