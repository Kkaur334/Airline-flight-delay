#!/usr/bin/env python
# coding: utf-8

# ## Importing Necessary Libariries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ## Reading the data

# In[2]:


flights_df = pd.read_csv("FlightDelays.csv")
print(flights_df)


# In[3]:


flights_df.shape


# ## Renaming the columns

# In[4]:


flights_df.columns = [s.strip().replace(' ', '_').upper() for s in flights_df.columns]
flights_df.columns


# In[5]:


## checking the null values


# In[6]:


flights_df.isnull().sum()


# ## Understanding of data

# In[7]:


flights_df.describe()


#  Now looking at the summary statistics, we will drop the row which has minimum value for 10 according to our analysis of being an incorrect value

# In[8]:


#Distributed of scheduled departure time and actual departure time
plt.figure(figsize=(5,5))
flights_df.boxplot(column=["CRS_DEP_TIME","DEP_TIME"])
plt.show()


# In[9]:


# Drop rows where DEP_TIME is 10
flights_df = flights_df[flights_df['DEP_TIME'] != 10]
flights_df.describe()


# In[10]:


#After dropping the row
#Distributed of scheduled departure time and actual departure time
plt.figure(figsize=(5,5))
flights_df.boxplot(column=["CRS_DEP_TIME","DEP_TIME"])
plt.show()


# In[11]:


flights_df.dtypes


# ## Analysis of the data

# In[12]:


# Delay rate by carrier
carrier_delay_rate = flights_df.groupby('CARRIER')['FLIGHT_STATUS'].value_counts(normalize=True).unstack()
carrier_delay_rate.plot(kind='bar', stacked=True)
plt.title('Delay Rate by Carrier')
plt.ylabel('Proportion')
plt.show()


# 1. The carriers MQ (likely American Eagle) and RU show the highest proportion of delayed flights among the carriers listed.
# 2. DL (Delta Air Lines) has the lowest proportion of delayed flights
# 3. There is a noticeable variation in delay proportions among carriers. Some carriers like MQ and RU have higher delays, while others like DL, UA and US  have lower delays.
# 

# In[13]:


# Bar plot for Flight Status
sns.countplot(x='FLIGHT_STATUS', data=flights_df)
plt.title('Flight Status Distribution')
plt.show()


# In our dataset, there are more ontime flights as compared to the delayed flights

# In[14]:


# Delay Rate by weather
carrier_delay_rate = flights_df.groupby('WEATHER')['FLIGHT_STATUS'].value_counts(normalize=True).unstack()
carrier_delay_rate.plot(kind='bar', stacked=True)
plt.title('Delay Rate by Weather')
plt.ylabel('Proportion')
plt.show()


# 1. Most flights are on time when the weather is clear. There is still a portion of flights that are delayed (about 80%) , but it is relatively small compared to the on-time flights.
# 2. All flights are delayed when the weather is adverse.

# In[15]:


#number of flights from each airport
flights_df['ORIGIN'].value_counts()


# Most of the flights are taking off from DCA terminal

# In[16]:


#to check the distribution of the flight status by the days of the week
grouped_data = flights_df.groupby(['DAY_WEEK', 'FLIGHT_STATUS']).size().unstack(fill_value=0)
grouped_data.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Number of Flights by Status and Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Flights')
plt.legend(title='Flight Status')
plt.xticks(rotation=45)
plt.show()


# 1. Day 5 (likely Friday) has the highest total number of flights, both on-time and delayed, compared to other days.
# 2. Day 7 (likely Sunday) has the lowest total number of flights.
# 3. In absolute numbers, Day 5 (Friday) also has the highest number of delayed flights due to its high total flight volume.
# 4. Day 6 (Saturday) has the lowest absolute number of delays, consistent with its lower total flight volume.

# In[17]:


#distrubution of flights across carrier
carrier_counts = flights_df['CARRIER'].value_counts()
sns.barplot(x=carrier_counts.index, y=carrier_counts.values)
plt.title('Number of flights per carrier')
plt.xlabel('Carrier')
plt.ylabel('Number of Flights')
plt.show()


# DH carrier has the highest number of flights taking off and OH and UA has the lowest number of flights taking off

# In[18]:


#delayed flights by carrier
delayed_flights = flights_df[flights_df['FLIGHT_STATUS'] == 'delayed']
carrier_delays = delayed_flights['CARRIER'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(carrier_delays, labels=carrier_delays.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
plt.title('Distribution of Delayed Flights by Carrier')
plt.show() 


# 1. The majority of delayed flights are concentrated among a few carriers, specifically DH, RU, and MQ, which together account for more than 70% of all delays.
# 2. The carriers with the smallest proportions of delayed flights (UA and OH) contribute less than 2% combined.

# ## Manipulation of data for the ease in analysis

# In[19]:


# changing the outcome variable column
flights_df['FLIGHT_STATUS']=np.where(flights_df['FLIGHT_STATUS'].str.contains('delayed'),1,0)
flights_df.head(5)


# ## Specificying the predictors and Outcomes, scaling the data 

# In[20]:


predictors=['DISTANCE','WEATHER','DAY_WEEK','DAY_OF_MONTH','CARRIER','DEST','ORIGIN']
X=pd.get_dummies(flights_df[predictors],drop_first=True)
X.head(5)


# In[21]:


Y = flights_df['FLIGHT_STATUS']
Y.head(5)


# In[22]:


scaler=StandardScaler()
scaled_fd=pd.DataFrame(scaler.fit_transform(X),index=X.index,columns=X.columns)
print(scaled_fd)


# ## Checking Skewness and correlation

# In[23]:


skewness=print(X.skew())


# 
# 1. WEATHER, CARRIER_OH, and CARRIER_UA have very high positive skewness, indicating that adverse weather and certain carriers have many lower values and few higher values.
# 2. DAY_WEEK and DAY_OF_MONTH are nearly symmetrical.
# 3. Most CARRIER and DEST variables exhibit positive skewness, suggesting these variables are not evenly distributed and have more frequent lower values with some higher values.

# In[24]:


scaled_fd.corr(). round(2)


# In[25]:


#combining the dataframe
combined_fd=pd.concat([scaled_fd,Y],axis=1)
corr=combined_fd.corr().round(3)
print(corr)


# In[26]:


#Plotting the heatmap to show the correlation
plt.figure(figsize=(9, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, annot_kws={"size": 10},
            linewidths=0.5, linecolor='gray')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.title("Correlation Heatmap", fontsize=15)
plt.tight_layout()
plt.show()


# 
# 1. Specific carriers, such as Carrier US (CARRIER_US) (-0.24) and Carrier DH (CARRIER_DH) (-0.18), show a notable negative correlation with delay minutes, suggesting flights operated by these carriers tend to experience fewer delays.
# 2. There is a strong positive correlation between flights originating from DCA (ORIGIN_DCA) and IAD (ORIGIN_IAD) (0.86), as well as between flights destined for JFK (DEST_JFK) and LGA (DEST_LGA) (0.47), indicating similar flight patterns or operational conditions between these pairs of airports.
# 

# In[27]:


correlation_Target=corr['FLIGHT_STATUS'].drop('FLIGHT_STATUS').round(3)
print(correlation_Target)


# Variables selected: "WEATHER","CARRIER_US","CARRIER_DL"

# ## 1. Fitting KNN MODEL

# In[28]:


#splitting the dataset
train_data,valid_data=train_test_split(combined_fd,test_size=0.4,random_state=1)
print(train_data.shape,valid_data.shape)


# In[29]:


train_data.head(5)


# In[30]:


knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(train_data[["WEATHER","CARRIER_US","CARRIER_DL"]],np.ravel(train_data[["FLIGHT_STATUS"]]))


# In[31]:


classification=knn.predict(valid_data[["WEATHER","CARRIER_US","CARRIER_DL"]])


# In[32]:


#checking the accuracy score
accuracy=accuracy_score(valid_data[["FLIGHT_STATUS"]],classification)
print(f'accuracy:{accuracy}')


# In[33]:


# Train a classifier for different values of k
results = []
for k in range(1, 15):
     knn = KNeighborsClassifier(n_neighbors=k).fit(train_data[["WEATHER","CARRIER_US","CARRIER_DL"]], train_data["FLIGHT_STATUS"])
     results.append({
    'k': k,
     'accuracy': accuracy_score(valid_data["FLIGHT_STATUS"], knn.predict(valid_data[["WEATHER","CARRIER_US","CARRIER_DL"]]))
})


# In[34]:


# Convert results to a pandas data frame
results = pd.DataFrame(results)
print(results)


# In[35]:


plt.plot(results["k"],results["accuracy"])


# Given the highest accuracy and stable performance around that region k=2, k=2 is the optimal choice as it provides the best accuracy without significant fluctuation.

# In[36]:


# predicting a new data point with  neighbours
new_data = pd.DataFrame([{"WEATHER":0,"CARRIER_US":True,"CARRIER_DL":False}])
print(new_data)


# In[37]:


#Train the KNN model on the training data. 
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(combined_fd[["WEATHER","CARRIER_US","CARRIER_DL"]],combined_fd["FLIGHT_STATUS"])
distances, indices = knn.kneighbors(new_data)
print(knn.predict(new_data))
print('Distances',distances)
print('Indices', indices)
print(combined_fd.iloc[indices[0], :])


# In[38]:


print("Accuracy of the KNN Model where k=2 is: ",round(results["accuracy"].max()*100,2),"%")


# ## 2. Fitting the Random Forest Model

# In[39]:


le = LabelEncoder()
categorical_features = ['CARRIER', 'DEST', 'ORIGIN', 'TAIL_NUM', 'FL_NUM']
for feature in categorical_features:
    flights_df[feature] = le.fit_transform(flights_df[feature])


# In[40]:


# Defining feature set (X) and target variable (y)
X = flights_df.drop(columns=['FLIGHT_STATUS','FL_DATE','CRS_DEP_TIME', 'DEP_TIME'])
y = flights_df['FLIGHT_STATUS']


# In[41]:


# Encoding the target variable
y = le.fit_transform(y)


# In[42]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)


# In[43]:


# Make predictions on the test set
y_pred = clf.predict(X_test)


# In[44]:


# Evaluate the model
accuracy_1 = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy_1:.2f}')


# In[45]:


# Print detailed classification report
print(classification_report(y_test, y_pred))


# In[46]:


#Trying Ensemble Methods to improve the performance
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier

# Example of stacking multiple models
stacked_model = VotingClassifier(estimators=[
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42))
], voting='soft')

stacked_model.fit(X_train, y_train)
y_pred_stack = stacked_model.predict(X_test)
print(classification_report(y_test, y_pred_stack))


# In[47]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_pred_stack)
plt.figure(figsize=(4, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# In[48]:


# Evaluate the model
accuracy_2 = accuracy_score(y_test, y_pred_stack)
print(f'Accuracy: {accuracy_2:.2f}')


# In[49]:


models = ['Random Forest', 'Ensemble Model']
accuracies = [accuracy_1, accuracy_2]

plt.figure()
plt.bar(models, accuracies, color=['blue', 'orange'])
plt.ylim([0, 1])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Random Forest vs Ensemble Model')
plt.show()


# In[ ]:




