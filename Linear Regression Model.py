#!/usr/bin/env python
# coding: utf-8

# ### Data Science and Business Analytics Intern at SPARKS Foundation

# #### Prediction using Supervised Machine Learning

# #### Author : Payal Gupta

# #### PROBLEM : Predict the percentage of a student based on the number of study hours.

# ### Importing and Reading the dataset

# In[23]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[24]:


# Reading Data from remote link
data= pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
print("Data Imported Successfully")
data.head(10) #prints top 10 rows of dataset


# In[25]:


data.shape


# #### The dataset contains 25 rows and 2 columns, i.e., Hours and scores

# In[26]:


data.describe() #describes the dataset using statistics


# In[27]:


data.info() #Prints a summary of columns count and its dtypes


# ### Visualisation of Data using Scatter plot 

# ##### Let's plot our data points on 2-D graph and see if we can manually find any relationship between the data. 

# In[30]:


#plotting the scores against number of hours
data.plot(x="Hours", y="Scores", style="o")
plt.title("Hours VS Scores")
plt.xlabel("No. of Hours")
plt.ylabel("Scores")
plt.show()


# #### From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# In[31]:


data.corr()


# #### We can say that both the variables are positively correlated to one another and a linear regression exists between them. 

# ### Splitting the Dataset 

# In[32]:


x=data.iloc[:, :-1].values #gives second last column of dataset
y=data.iloc[:,1].values #gives last column of dataset


# ### Splitting dataset into training and test sets

# #### We'll do this by using Scikit-Learn's built-in train_test_split() method: 

# In[33]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# ### Training the Algorithm 

# In[34]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print("Score: ",regressor.score(x_train,y_train))
print("Training Complete.")


# ### Plotting the Regression Line 

# In[39]:


#evaluating coefficient and intercept
m=regressor.coef_
c=regressor.intercept_
print("Coefficient: ",m)
print("Intercept: ",c)


# In[43]:


#plotting the regression line
line=m*x+c

#plotting for the test data
plt.scatter(x,y)
plt.plot(x,line,"orange")
plt.title("Hours VS scores")
plt.xlabel("No.of Hours")
plt.ylabel("Scores")
plt.show()


# ### Making Predictions

# In[44]:


print("Testing dataset - in hours\n",x_test) #testing data in hours


# In[45]:


y_pred=regressor.predict(x_test) #predicting the scores
print("Predicted y:\n",y_pred)


# ### Comparing Actual VS Predicted values

# In[49]:


data =pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
data


# ### Predicting the score for 9.25 hours/day

# In[54]:


#you can also test your own data
hours=9.25
own_pred=regressor.predict([[hours],])
print("No. of hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# #### If a student studies 9.25 hrs/day, Linear Regression Model prediction says that the student will score 93.69%

# ### Evaluating the Model

# #### As we are dealing with numeric data, Mean absolute error, Mean squared error would be a great choice for a metric for evaluating the Linear Regression Model.

# In[56]:


from sklearn import metrics
print("Mean Absolute Error :",metrics.mean_absolute_error(y_test,y_pred))
print("Mean Squared Error :",metrics.mean_squared_error(y_test,y_pred))


# In[57]:


accuracy=regressor.score(x_test,y_test)
print("Accuracy :",accuracy*100,"%")

