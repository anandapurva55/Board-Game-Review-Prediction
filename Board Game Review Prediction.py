# In[33]:

import sys
import matplotlib
import pandas
import sklearn
import seaborn


# In[34]:


print(pandas.__version__)


# In[35]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[36]:


games = pandas.read_csv("games.csv")
print(games.shape)
print(games.columns)


# In[37]:


plt.hist(games["average_rating"])
plt.show()


# In[38]:


#Print the first row with score = 0
print(games[games["average_rating"] == 0].iloc[0])

#Print the first row with score > 0
print(games[games["average_rating"] > 0].iloc[0])


# In[39]:


#Remove any row without user review
games = games[games["users_rated"] > 0]

#Remove any row with missing values
games = games.dropna(axis=0)

#Make a histogram of all average ratings
plt.hist(games["average_rating"])
plt.show()


# In[40]:


#coorelation matrix

corrmat = games.corr() #part of pandas data frame

fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat, vmax = .8 , square = True)
plt.show()


# In[41]:


#Get all the columns from the dataFrame
columns = games.columns.tolist()


#Filter the columns from the data we don't want
columns = [c for c in columns if c not in ["bayes_average_rating","average_rating","type","name","id"]]

#Store the variables we will be predicting on
target = "average_rating"



# In[42]:


#Generate training and test datasets
from sklearn.model_selection import train_test_split


#Generate training set
train = games.sample(frac = 0.8, random_state = 1)

#Select anything not in the training set and put in the test
test = games.loc[~games.index.isin(train.index)]

print(train.shape)
print(test.shape)


# In[43]:


#Import Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Initialize the model class
LR = LinearRegression()
#Get all the columns from the dataFrame
columns = games.columns.tolist()


#Filter the columns from the data we don't want
columns = [c for c in columns if c not in ["bayes_average_rating","average_rating","type","name","id"]]
target = "average_rating"
#Fit the medel the training data
LR.fit(train[columns],train[target])


# In[44]:


#Generate predictions for the test set
predictions = LR.predict(test[columns])

#Compute error between our test prediction and actual values
mean_squared_error(predictions,test[target])


# In[49]:


#import the forest regressor model
from sklearn.ensemble import RandomForestRegressor

#Initialize the model
RFR = RandomForestRegressor(n_estimators = 100, min_samples_leaf = 10, random_state = 1)

#Fit to the data
RFR.fit(train[columns],train[target])


# In[57]:


#make predictions
predictions = RFR.predict(test[columns])

#compute the error between our text predictions and actual values
mean_squared_error(predictions, test[target])


# In[58]:


test[columns].iloc[90]


# In[59]:


#Make predictions with both samples
rating_LR = LR.predict(test[columns].iloc[90].values.reshape(1,-1))
rating_RFR = RFR.predict(test[columns].iloc[90].values.reshape(1,-1))

#Print out the prediction
print(rating_LR)
print(rating_RFR)







