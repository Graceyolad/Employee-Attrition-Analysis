#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
#sns.set_theme(style = 'white')
#sns.set_theme(style = 'darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')


#import time
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import train_test_split, GridSearchCV


#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC


#from sklearn.metrics import classification_report
#from sklearn import metrics
#from sklearn.metrics import confusion_matrix

import warnings;
warnings.filterwarnings('ignore');





# In[3]:


#from matplotlib import style
#style.use('dark_background')


# In[4]:


#!pip install jupyterthemes 
#!jt -t onedork
#from jupyterthemes import jtplot
#jtplot.style(theme ='monokai', context = 'notebook', ticks = True, grid = False)


# In[5]:


ewl = pd.read_csv('ewl.csv', index_col = 0) #specifying the index 0 makes the EmpID the index
ewl


# In[6]:


ee = pd.read_csv('ee.csv', index_col = 0)
ee


# Data Exploration

# In[7]:


ee.shape


# In[8]:


ewl.shape


# In[9]:


ee.columns


# In[10]:


ewl.columns


# In[11]:


#adding a new column 'Attrition for existing employees'
ee['Attrition'] = 'No'
ee


# In[12]:


#adding a new column 'Attrition for employees who left' 
ewl['Attrition'] = 'Yes'
ewl


# In[13]:


#joining the two DataFrames
data = pd.concat([ewl,ee])
data


# In[14]:


data.info()


# In[15]:


data.nunique()


# In[16]:


#checking data types
data.dtypes


# In[17]:


#attrition unique values
data.Attrition.value_counts()/len(data)*100


# Key points about the dataset
# 1. There are 14999 observations and 10 features
# 2. The dataset contains data types float, integer and object
# 3. The data contains no missing values
# 4. Attrition is the target variable in this dataset while others  are independent variables.
# 5. About 24% of employees left the company while 76% remained.
# 

# Transforming target variable to numeric values(0,1)

# In[18]:


#changing target variable  'Attrition' to 1 as 'Yes' and 0 as 'No'
data['Attrition'] = np.where(data['Attrition'] == 'Yes', 1, 0)


# In[19]:


data


# ## EDA and Visualization

# ### Attrition

# In[20]:


#setting color palette
col_pal = sns.color_palette('GnBu' , n_colors = 2)


# In[21]:


#pie plot
labels = 'No', 'Yes'
sizes = (data.Attrition.value_counts()/len(data)*100)
explode = (0.1, 0) # only explode first size
fig1,ax1 = plt.subplots()
ax1.pie(sizes, explode = explode, labels = labels, autopct = '%0.1f%%', shadow = True, startangle = 90, colors = col_pal)
ax1.axis('equal') # ensures that the pie is drawn as a circle.
plt.title('Attrition among Employees')
plt.show()


# In[22]:


sns.set_theme(style = 'white')
ax = sns.countplot(x = 'Attrition', data = data,  hue = 'salary', palette = 'Set3', saturation = 1)
plt.title('Employee Attrition by Salary')
plt.xlabel('Employee Attrition')
plt.ylabel('Number of Employees')

for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()))



# From the above plot, about 60% of the employees who left have a low salary, followed by medium salary (about 37%).
# Only about 2% of employees with high salary left.

# In[23]:


#setting dimensions of the plot

sns.set(rc= {'figure.figsize' : (12,8)}) #width 8, height 4
ax = sns.countplot(x = 'dept', data = data,  palette = 'Set3', saturation = 1)
plt.title('Number of Employees per Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')



for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()))


# In[24]:


sns.set_theme(style = 'white')
sns.set(rc= {'figure.figsize' : (15,8)}) #width 8, height 4
sns.countplot(x = 'dept', data = data,  hue = 'salary', palette = 'Set3', saturation = 1)
plt.title('Number of Employees by Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')




# In[25]:


#defining a function to show percentages 
def perct(ax,feature):
    total = len(feature)
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100* p.get_height()/total)
        x = p.get_x() + p.get_width()/2-0.05
        y = p.get_y() + p.get_height()
        ax.annotate(percentage, (x,y), size = 12)


# In[26]:


#displaying plots
ax = sns.countplot(x = 'dept', data = data, hue = 'Attrition', palette = 'Set3', saturation = 1)
plt.title('Percentage of Employees per Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')


perct(ax, data.dept)


# In[27]:


#setting dimensions of the plot

sns.set(rc= {'figure.figsize' : (12,8)}) #width 8, height 4
ax = sns.countplot(x = 'dept', data = data, hue = 'Attrition', palette = 'Set3', saturation = 1)
plt.title('Count of Employees per Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')



for p in ax.patches:
    ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.1, p.get_height()))


# As shown in the above plots, considering the total employees in the organization, Management department has the lowest percentage of employees who left (0.6%), with the highest percentage in Sales (6.8%).
# However, considering the number of employees who left per department, sales department has the highest number of employees.
# 29% of employees in HR left, followed by Accounting(27%), Technical (26%), Support (25%), Sales and Marketing (24%), IT and Product_mng(22%), RandD (15%), Management(14%).

# In[28]:


#showing the correlation of the target variable with other variables
corr = data.corr()
plt.figure(figsize=(12,12))
sns.heatmap(corr,cbar=True,square=True,fmt='.1f',annot=True,cmap='Reds')


# ### Uni Variate Numerical Feature Analysis

# In[29]:


fig, axes = plt.subplots(2,3, figsize = (16,10))
sns.distplot(data['satisfaction_level'], ax = axes[0,0], color = 'lightseagreen')
sns.distplot(data['last_evaluation'], ax = axes[0,1], color = 'steelblue')
sns.distplot(data['number_project'], ax = axes[0,2], color = 'lightseagreen')
sns.distplot(data['average_montly_hours'], ax = axes[1,0], color = 'steelblue')
sns.distplot(data['time_spend_company'], ax = axes[1,1], color = 'lightseagreen')
sns.distplot(data['Work_accident'], ax = axes[1,2], color = 'lightseagreen')

plt.show()


# From the above plots, outliers are observed in satisfaction_level, number_project, time_spend_company and work_accident

# ### Bi Variate Categorical Feature Analysis

# In[30]:


data.head()


# In[32]:


mean_sat_dept = data.groupby(['dept', 'Attrition'])['satisfaction_level'].mean().to_frame()
mean_sat_dept


# In[33]:


mean_sat = data.groupby(['Attrition'])['satisfaction_level'].mean().to_frame()
mean_sat


#  From the above table, the mean satisfaction level could be a factor contributing to Attrition. The mean satisfactory level of Employees who left in each department is less than that of Existing employees

# The Employees who Attrited have a lower satisfaction level compared to existing employees

# In[34]:


data.groupby(['dept', 'Attrition'])['last_evaluation'].mean().to_frame()


# In[35]:


sns.boxplot(x = 'Attrition', y = 'last_evaluation', data = data)
plt.title('Last Evaluation')


# In[36]:


data.groupby(['Attrition'])['last_evaluation'].mean().to_frame()


# surprisingly, even though employees who left generally have a low salary, they have almost the same evaluation with the existing employees which further corroborates the fact that they were underpaid

# In[37]:


data.groupby(['dept', 'Attrition'])['number_project'].mean().to_frame()


# In[38]:


data.groupby(['Attrition'])['number_project'].mean().to_frame()


# In[39]:


sns.countplot('number_project', hue = 'Attrition', data=data)


# Employees who left also carried out a higher number of projects even though they had a lower salary

# In[40]:


data.groupby(['dept', 'Attrition'])['average_montly_hours'].mean().to_frame()


# In[41]:


sns.boxplot(x = 'Attrition', y = 'average_montly_hours', data = data)
plt.title('Average Monthly Hours by Attrition')


# In[42]:


data.groupby(['Attrition'])['average_montly_hours'].mean().to_frame()

The Employees who left worked for more hours monthly compared to Existing Employees
# In[37]:


data.groupby(['dept', 'Attrition'])['time_spend_company'].mean().to_frame()


# In[38]:


sns.boxplot(x = 'Attrition', y = 'time_spend_company', data = data)


# In[41]:


data.groupby(['Attrition'])['time_spend_company'].mean().to_frame()


# Employees who left had been in the company for longer

# In[39]:


data.groupby(['dept', 'Attrition'])['Work_accident'].mean().to_frame()


# In[42]:


data.groupby(['Attrition'])['Work_accident'].mean().to_frame()


# In[40]:


corr_data = data.corr()
corr_data


# Converting categorical Features

# In[41]:



Dept = pd.get_dummies(data['dept'], drop_first = True)
Salary = pd.get_dummies(data['salary'], drop_first = True)


# In[42]:


Dept.head()


# In[43]:


Salary.head()


# In[44]:


data.drop(['dept', 'salary'], axis = 1, inplace = True)


# In[45]:


data.head()


# In[46]:


data= pd.concat([data,Dept, Salary], axis = 1)


# In[47]:


data.head()


# Building a Logistic Regression Model

# In[48]:


#splitting data into test and train sets
#train test split
from sklearn.model_selection import train_test_split


# In[49]:


x = data [['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'RandD', 'accounting', 'hr', 'management', 'marketing', 'product_mng','sales', 'support', 'technical', 'low','medium']]
y = data['Attrition']


# In[50]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 101)


# Training and Predicting

# In[51]:


from sklearn.linear_model import LogisticRegression


# In[52]:


logmodel = LogisticRegression(solver = 'lbfgs', max_iter = 1000)
#logmodel = LogisticRegression()
logmodel.fit(x_train, y_train)


# In[53]:


prediction = logmodel.predict(x_test)


# Evaluation

# In[54]:


from sklearn.metrics import classification_report


# In[55]:


print(classification_report(y_test, prediction))


# In[56]:


from sklearn import metrics


# In[57]:


cnf_matrix = metrics.confusion_matrix(y_test, prediction)
cnf_matrix


# In[58]:


#r = df.groupby('Attrition')['Attrition'].count()
#plt.pie(r, explode=[0.05, 0.1], labels=['No', 'Yes'], radius=1.5, autopct='%1.1f%%',  shadow=True);


# Summary

# 24% of employees left the company
# Based on this dataset, some reasons why employees may have left include:
# 1. Low salary (60% of employees who left have a low salary)
# 2. Low satisfaction level. Employees who left have a low satisfaction level of about 0.4 while those who remained have a higher satisfaction level of about 0.7.
# 3. Higher average monthly hours. Employees who left have a higher average monthly hours (207) compared to employees who remained (199)
# 
# 
