#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression with Scikit Learn - Machine Learning with Python

# The following topics are covered in this tutorial:
# 
# - Exploratory data analysis and visualization
# - Splitting a dataset into training, validation & test sets
# - Filling/imputing missing values in numeric columns
# - Scaling numeric features to a $(0,1)$ range
# - Encoding categorical columns as one-hot vectors
# - Training a logistic regression model using Scikit-learn
# - Evaluating a model using a validation set and test set
# - Saving a model to disk and loading it back
# 

# ## Problem Statement
# 
# This tutorial takes a practical and coding-focused approach. We'll learn how to apply _logistic regression_ to a real-world dataset from [Kaggle](https://kaggle.com/datasets):
# 
# > **QUESTION**: The [Rain in Australia dataset](https://kaggle.com/jsphyg/weather-dataset-rattle-package) contains about 10 years of daily weather observations from numerous Australian weather stations. Here's a small sample from the dataset:
# > 
# > ![](https://i.imgur.com/5QNJvir.png)
# >
# > As a data scientist at the Bureau of Meteorology, you are tasked with creating a fully-automated system that can use today's weather data for a given location to predict whether it will rain at the location tomorrow. 
# >
# >
# > ![](https://i.imgur.com/KWfcpcO.png)
# 


# Classification problems can be binary (yes/no) or multiclass (picking one of many classes).
# 

# ### Regression Problems
# 
# Problems where a continuous numeric value must be predicted for each input are known as _regression problems_.
# 
# Here are some example of regression problems:
# 
# - [Medical Charges Prediction](https://www.kaggle.com/subhakarks/medical-insurance-cost-analysis-and-prediction)
# - [House Price Prediction](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) 
# - [Ocean Temperature Prediction](https://www.kaggle.com/sohier/calcofi)
# - [Weather Temperature Prediction](https://www.kaggle.com/budincsevity/szeged-weather)

# ### Logistic Regression for Solving Classification Problems
# 
# Logistic regression is a commonly used technique for solving binary classification problems. In a logistic regression model: 
# 
# - we take linear combination (or weighted sum of the input features) 
# - we apply the sigmoid function to the result to obtain a number between 0 and 1
# - this number represents the probability of the input being classified as "Yes"
# - instead of RMSE, the cross entropy loss function is used to evaluate the results
# 


# ### Machine Learning Workflow
# 
# Whether we're solving a regression problem using linear regression or a classification problem using logistic regression, the workflow for training a model is exactly the same:
# 
# 1. We initialize a model with random parameters (weights & biases).
# 2. We pass some inputs into the model to obtain predictions.
# 3. We compare the model's predictions with the actual targets using the loss function.  
# 4. We use an optimization technique (like least squares, gradient descent etc.) to reduce the loss by adjusting the weights & biases of the model
# 5. We repeat steps 1 to 4 till the predictions from the model are good enough.
# 

# 
# Let's load the data from `weatherAUS.csv` using Pandas.

# In[1]:


import pandas as pd


# In[4]:


raw_df = pd.read_csv('weatherAUS.csv')


# In[5]:


raw_df


# The dataset contains over 145,000 rows and 23 columns. The dataset contains date, numeric and categorical columns. Our objective is to create a model to predict the value in the column `RainTomorrow`.
# 
# Let's check the data types and missing values in the various columns.

# In[6]:


raw_df.info()


# While we should be able to fill in missing values for most columns, it might be a good idea to discard the rows where the value of `RainTomorrow` or `RainToday` is missing to make our analysis and modeling simpler (since one of them is the target variable, and the other is likely to be very closely related to the target variable). 

# In[7]:


raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)


# ## Exploratory Data Analysis and Visualization
# 
# Before training a machine learning model, its always a good idea to explore the distributions of various columns and see how they are related to the target column. Let's explore and visualize the data using the Plotly, Matplotlib and Seaborn libraries. 

# In[8]:


# let's install plotly
get_ipython().system('pip install plotly')


# In[9]:


import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[10]:


px.histogram(raw_df, x='Location', title='Location vs. Rainy Days', color='RainToday')


# In[11]:


px.histogram(raw_df, 
             x='Temp3pm', 
             title='Temperature at 3 pm vs. Rain Tomorrow', 
             color='RainTomorrow')


# In[12]:


px.histogram(raw_df, 
             x='RainTomorrow', 
             color='RainToday', 
             title='Rain Tomorrow vs. Rain Today')


# In[13]:


px.scatter(raw_df.sample(2000), 
           title='Min Temp. vs Max Temp.',
           x='MinTemp', 
           y='MaxTemp', 
           color='RainToday')


# In[14]:


px.scatter(raw_df.sample(2000), 
           title='Temp (3 pm) vs. Humidity (3 pm)',
           x='Temp3pm',
           y='Humidity3pm',
           color='RainTomorrow')


# ## Training, Validation and Test Sets
# 
# While building real-world machine learning models, it is quite common to split the dataset into three parts:
# 
# 1. **Training set** - used to train the model, i.e., compute the loss and adjust the model's weights using an optimization technique. 
# 
# 
# 2. **Validation set** - used to evaluate the model during training, tune model hyperparameters (optimization technique, regularization etc.), and pick the best version of the model. Picking a good validation set is essential for training models that generalize well. [Learn more here.](https://www.fast.ai/2017/11/13/validation-sets/)
# 
# 
# 3. **Test set** - used to compare different models or approaches and report the model's final accuracy. For many datasets, test sets are provided separately. The test set should reflect the kind of data the model will encounter in the real-world, as closely as feasible.
# 
# 
# <img src="https://i.imgur.com/j8eITrK.png" width="480">
# 
# 
# As a general rule of thumb you can use around 60% of the data for the training set, 20% for the validation set and 20% for the test set. If a separate test set is already provided, you can use a 75%-25% training-validation split.
# 
# 
# When rows in the dataset have no inherent order, it's common practice to pick random subsets of rows for creating test and validation sets. This can be done using the `train_test_split` utility from `scikit-learn`. Learn more about it here: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


train_val_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=42)


# In[17]:


print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)


# However, while working with dates, it's often a better idea to separate the training, validation and test sets with time, so that the model is trained on data from the past and evaluated on data from the future.
# 
# For the current dataset, we can use the Date column in the dataset to create another column for year. We'll pick the last two years for the test set, and one year before it for the validation set.

# In[18]:


plt.title('No. of Rows per Year')
sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year);


# In[19]:


year = pd.to_datetime(raw_df.Date).dt.year

train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]


# In[20]:


print('train_df.shape :', train_df.shape)
print('val_df.shape :', val_df.shape)
print('test_df.shape :', test_df.shape)


# While not a perfect 60-20-20 split, we have ensured that the test validation and test sets both contain data for all 12 months of the year.

# In[21]:


train_df


# In[22]:


val_df


# In[23]:


test_df


# ## Identifying Input and Target Columns
# 
# Often, not all the columns in a dataset are useful for training a model. In the current dataset, we can ignore the `Date` column, since we only want to weather conditions to make a prediction about whether it will rain the next day.
# 
# Let's create a list of input columns, and also identify the target column.

# In[24]:


input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'


# In[25]:


print(input_cols)


# In[26]:


target_col


# We can now create inputs and targets for the training, validation and test sets for further processing and model training.

# In[27]:


train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()


# In[28]:


val_inputs = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()


# In[29]:


test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()


# In[30]:


train_inputs


# In[31]:


train_targets


# Let's also identify which of the columns are numerical and which ones are categorical. This will be useful later, as we'll need to convert the categorical data to numbers for training a logistic regression model.

# In[32]:


import numpy as np


# In[33]:


numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()[:-1]
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()


# Let's view some statistics for the numeric columns.

# In[34]:


train_inputs[numeric_cols].describe()


# Do the ranges of the numeric columns seem reasonable? If not, we may have to do some data cleaning as well.
# 
# Let's also check the number of categories in each of the categorical columns.

# In[35]:


train_inputs[categorical_cols].nunique()


# Let's save our work before continuing.

# ## Imputing Missing Numeric Data
# 
# Machine learning models can't work with missing numerical data. The process of filling missing values is called imputation.
# 
# <img src="https://i.imgur.com/W7cfyOp.png" width="480">
# 
# There are several techniques for imputation, but we'll use the most basic one: replacing missing values with the average value in the column using the `SimpleImputer` class from `sklearn.impute`.

# In[36]:


from sklearn.impute import SimpleImputer


# In[37]:


imputer = SimpleImputer(strategy = 'mean')


# Before we perform imputation, let's check the no. of missing values in each numeric column.

# In[38]:


raw_df[numeric_cols].isna().sum()


# These values are spread across the training, test and validation sets. You can also check the no. of missing values individually for `train_inputs`, `val_inputs` and `test_inputs`.

# In[39]:


train_inputs[numeric_cols].isna().sum()


# The first step in imputation is to `fit` the imputer to the data i.e. compute the chosen statistic (e.g. mean) for each column in the dataset. 

# In[40]:


imputer.fit(raw_df[numeric_cols])


# After calling `fit`, the computed statistic for each column is stored in the `statistics_` property of `imputer`.

# In[41]:


list(imputer.statistics_)


# The missing values in the training, test and validation sets can now be filled in using the `transform` method of `imputer`.

# In[42]:


train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])


# The missing values are now filled in with the mean of each column.

# In[43]:


train_inputs[numeric_cols].isna().sum()


# > **EXERCISE**: Apply some other imputation techniques and observe how they change the results of the model. You can learn more about other imputation techniques here: https://scikit-learn.org/stable/modules/impute.html

# ## Scaling Numeric Features
# 
# Another good practice is to scale numeric features to a small range of values e.g. $(0,1)$ or $(-1,1)$. Scaling numeric features ensures that no particular feature has a disproportionate impact on the model's loss. Optimization algorithms also work better in practice with smaller numbers.
# 
# The numeric columns in our dataset have varying ranges.

# In[44]:


raw_df[numeric_cols].describe()


# Let's use `MinMaxScaler` from `sklearn.preprocessing` to scale values to the $(0,1)$ range.

# In[45]:


from sklearn.preprocessing import MinMaxScaler


# In[46]:


get_ipython().run_line_magic('pinfo', 'MinMaxScaler')


# In[47]:


scaler = MinMaxScaler()


# First, we `fit` the scaler to the data i.e. compute the range of values for each numeric column.

# In[48]:


scaler.fit(raw_df[numeric_cols])


# We can now inspect the minimum and maximum values in each column.

# In[49]:


print('Minimum:')
list(scaler.data_min_)


# In[50]:


print('Maximum:')
list(scaler.data_max_)


# We can now separately scale the training, validation and test sets using the `transform` method of `scaler`.

# In[51]:


train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])


# We can now verify that values in each column lie in the range $(0,1)$

# In[52]:


train_inputs[numeric_cols].describe()


# Learn more about scaling techniques here: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/

# Let's save our work before continuing.

# ## Encoding Categorical Data
# 
# Since machine learning models can only be trained with numeric data, we need to convert categorical data to numbers. A common technique is to use one-hot encoding for categorical columns.
# 
# <img src="https://i.imgur.com/n8GuiOO.png" width="640">
# 
# One hot encoding involves adding a new binary (0/1) column for each unique category of a categorical column. 

# In[53]:


raw_df[categorical_cols].nunique()


# We can perform one hot encoding using the `OneHotEncoder` class from `sklearn.preprocessing`.

# In[54]:


from sklearn.preprocessing import OneHotEncoder


# In[54]:


get_ipython().run_line_magic('pinfo', 'OneHotEncoder')


# In[55]:


encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')


# First, we `fit` the encoder to the data i.e. identify the full list of categories across all categorical columns.

# In[56]:


encoder.fit(raw_df[categorical_cols])


# In[ ]:


encoder.categories_


# The encoder has created a list of categories for each of the categorical columns in the dataset. 
# 
# We can generate column names for each individual category using `get_feature_names`.

# In[ ]:


encoded_cols = list(encoder.get_feature_names(categorical_cols))
print(encoded_cols)


# All of the above columns will be added to `train_inputs`, `val_inputs` and `test_inputs`.
# 
# To perform the encoding, we use the `transform` method of `encoder`.

# In[ ]:


train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])


# We can verify that these new columns have been added to our training, test and validation sets.

# In[ ]:


pd.set_option('display.max_columns', None)


# In[ ]:


test_inputs


# ## Training a Logistic Regression Model
# 
# Logistic regression is a commonly used technique for solving binary classification problems. In a logistic regression model: 
# 
# - we take linear combination (or weighted sum of the input features) 
# - we apply the sigmoid function to the result to obtain a number between 0 and 1
# - this number represents the probability of the input being classified as "Yes"
# - instead of RMSE, the cross entropy loss function is used to evaluate the results
# 
# 
# Here's a visual summary of how a logistic regression model is structured ([source](http://datahacker.rs/005-pytorch-logistic-regression-in-pytorch/)):
# 
# 
# <img src="https://i.imgur.com/YMaMo5D.png" width="480">
# 
# The sigmoid function applied to the linear combination of inputs has the following formula:
# 
# <img src="https://i.imgur.com/sAVwvZP.png" width="400">
# 
# To train a logistic regression model, we can use the `LogisticRegression` class from Scikit-learn.

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


get_ipython().run_line_magic('pinfo', 'LogisticRegression')


# In[ ]:


model = LogisticRegression(solver='liblinear')


# We can train the model using `model.fit`.

# In[ ]:


model.fit(train_inputs[numeric_cols + encoded_cols], train_targets)


# `model.fit` uses the following workflow for training the model ([source](https://www.deepnetts.com/blog/from-basic-machine-learning-to-deep-learning-in-5-minutes.html)):
# 
# 1. We initialize a model with random parameters (weights & biases).
# 2. We pass some inputs into the model to obtain predictions.
# 3. We compare the model's predictions with the actual targets using the loss function.  
# 4. We use an optimization technique (like least squares, gradient descent etc.) to reduce the loss by adjusting the weights & biases of the model
# 5. We repeat steps 1 to 4 till the predictions from the model are good enough.
# 
# 
# <img src="https://www.deepnetts.com/blog/wp-content/uploads/2019/02/SupervisedLearning.png" width="480">
# 
# For a mathematical discussion of logistic regression, sigmoid activation and cross entropy, check out [this YouTube playlist](https://www.youtube.com/watch?v=-la3q9d7AKQ&list=PLNeKWBMsAzboR8vvhnlanxCNr2V7ITuxy&index=1). Logistic regression can also be applied to multi-class classification problems, with a few modifications.
# 

# Let's check the weights and biases of the trained model.

# In[ ]:


print(numeric_cols + encoded_cols)


# In[ ]:


print(model.coef_.tolist())


# In[ ]:


print(model.intercept_)


# Each weight is applied to the value in a specific column of the input. Higher the weight, greater the impact of the column on the prediction.

# ## Making Predictions and Evaluating the Model
# 
# We can now use the trained model to make predictions on the training, test 

# In[ ]:


X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]


# In[ ]:


train_preds = model.predict(X_train)


# In[ ]:


train_preds


# In[ ]:


train_targets


# We can output a probabilistic prediction using `predict_proba`.

# In[ ]:


train_probs = model.predict_proba(X_train)
train_probs


# The numbers above indicate the probabilities for the target classes "No" and "Yes".

# In[ ]:


model.classes_


# We can test the accuracy of the model's predictions by computing the percentage of matching values in `train_preds` and `train_targets`.
# 
# This can be done using the `accuracy_score` function from `sklearn.metrics`.

# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(train_targets, train_preds)


# The model achieves an accuracy of 85.1% on the training set. We can visualize the breakdown of correctly and incorrectly classified inputs using a confusion matrix.
# 
# <img src="https://i.imgur.com/UM28BCN.png" width="480">

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(train_targets, train_preds, normalize='true')


# Let's define a helper function to generate predictions, compute the accuracy score and plot a confusion matrix for a given st of inputs.

# In[ ]:


def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)
    
    accuracy = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    
    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name));
    
    return preds


# In[ ]:


train_preds = predict_and_plot(X_train, train_targets, 'Training')


# Let's compute the model's accuracy on the validation and test sets too.

# In[ ]:


val_preds = predict_and_plot(X_val, val_targets, 'Validatiaon')


# In[ ]:


test_preds = predict_and_plot(X_test, test_targets, 'Test')


# The accuracy of the model on the test and validation set are above 84%, which suggests that our model generalizes well to data it hasn't seen before. 
# 
# But how good is 84% accuracy? While this depends on the nature of the problem and on business requirements, a good way to verify whether a model has actually learned something useful is to compare its results to a "random" or "dumb" model.
# 
# Let's create two models: one that guesses randomly and another that always return "No". Both of these models completely ignore the inputs given to them.

# In[ ]:


def random_guess(inputs):
    return np.random.choice(["No", "Yes"], len(inputs))


# In[ ]:


def all_no(inputs):
    return np.full(len(inputs), "No")


# Let's check the accuracies of these two models on the test set.

# In[ ]:


accuracy_score(test_targets, random_guess(X_test))


# In[ ]:


accuracy_score(test_targets, all_no(X_test))


# Our random model achieves an accuracy of 50% and our "always No" model achieves an accuracy of 77%. 
# 
# Thankfully, our model is better than a "dumb" or "random" model! This is not always the case, so it's a good practice to benchmark any model you train against such baseline models.

# ## Making Predictions on a Single Input
# 
# Once the model has been trained to a satisfactory accuracy, it can be used to make predictions on new data. Consider the following dictionary containing data collected from the Katherine weather department today.

# In[ ]:


new_input = {'Date': '2021-06-19',
             'Location': 'Katherine',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}


# The first step is to convert the dictionary into a Pandas dataframe, similar to `raw_df`. This can be done by passing a list containing the given dictionary to the `pd.DataFrame` constructor.

# In[ ]:


new_input_df = pd.DataFrame([new_input])


# In[ ]:


new_input_df


# We've now created a Pandas dataframe with the same columns as `raw_df` (except `RainTomorrow`, which needs to be predicted). The dataframe contains just one row of data, containing the given input.
# 
# 
# We must now apply the same transformations applied while training the model:
# 
# 1. Imputation of missing values using the `imputer` created earlier
# 2. Scaling numerical features using the `scaler` created earlier
# 3. Encoding categorical features using the `encoder` created earlier

# In[ ]:


new_input_df[numeric_cols] = imputer.transform(new_input_df[numeric_cols])
new_input_df[numeric_cols] = scaler.transform(new_input_df[numeric_cols])
new_input_df[encoded_cols] = encoder.transform(new_input_df[categorical_cols])


# In[ ]:


X_new_input = new_input_df[numeric_cols + encoded_cols]
X_new_input


# We can now make a prediction using `model.predict`.

# In[ ]:


prediction = model.predict(X_new_input)[0]


# In[ ]:


prediction


# Our model predicts that it will rain tomorrow in Katherine! We can also check the probability of the prediction.

# In[ ]:


prob = model.predict_proba(X_new_input)[0]


# In[ ]:


prob


# Looks like our model isn't too confident about its prediction!

# Let's define a helper function to make predictions for individual inputs.

# In[ ]:


def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob


# We can now use this function to make predictions for individual inputs.

# In[ ]:


new_input = {'Date': '2021-06-19',
             'Location': 'Launceston',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}


# In[ ]:


predict_input(new_input)


# In[ ]:


raw_df.Location.unique()


# ## Putting it all Together
# 
# While we've covered a lot of ground in this tutorial, the number of lines of code for processing the data and training the model is fairly small. Each step requires no more than 3-4 lines of code.

# ### Data Preprocessing

# In[ ]:


import opendatasets as od
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Download the dataset
od.download('https://www.kaggle.com/jsphyg/weather-dataset-rattle-package')
raw_df = pd.read_csv('weather-dataset-rattle-package/weatherAUS.csv')
raw_df.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)

# Create training, validation and test sets
year = pd.to_datetime(raw_df.Date).dt.year
train_df, val_df, test_df = raw_df[year < 2015], raw_df[year == 2015], raw_df[year > 2015]

# Create inputs and targets
input_cols = list(train_df.columns)[1:-1]
target_col = 'RainTomorrow'
train_inputs, train_targets = train_df[input_cols].copy(), train_df[target_col].copy()
val_inputs, val_targets = val_df[input_cols].copy(), val_df[target_col].copy()
test_inputs, test_targets = test_df[input_cols].copy(), test_df[target_col].copy()

# Identify numeric and categorical columns
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()[:-1]
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

# Impute missing numerical values
imputer = SimpleImputer(strategy = 'mean').fit(raw_df[numeric_cols])
train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = imputer.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

# Scale numeric features
scaler = MinMaxScaler().fit(raw_df[numeric_cols])
train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_inputs[numeric_cols] = scaler.transform(val_inputs[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(raw_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names(categorical_cols))
train_inputs[encoded_cols] = encoder.transform(train_inputs[categorical_cols])
val_inputs[encoded_cols] = encoder.transform(val_inputs[categorical_cols])
test_inputs[encoded_cols] = encoder.transform(test_inputs[categorical_cols])

# Save processed data to disk
train_inputs.to_parquet('train_inputs.parquet')
val_inputs.to_parquet('val_inputs.parquet')
test_inputs.to_parquet('test_inputs.parquet')
pd.DataFrame(train_targets).to_parquet('train_targets.parquet')
pd.DataFrame(val_targets).to_parquet('val_targets.parquet')
pd.DataFrame(test_targets).to_parquet('test_targets.parquet')

# Load processed data from disk
train_inputs = pd.read_parquet('train_inputs.parquet')
val_inputs = pd.read_parquet('val_inputs.parquet')
test_inputs = pd.read_parquet('test_inputs.parquet')
train_targets = pd.read_parquet('train_targets.parquet')[target_col]
val_targets = pd.read_parquet('val_targets.parquet')[target_col]
test_targets = pd.read_parquet('test_targets.parquet')[target_col]


# ### Model Training and Evaluation
# 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Select the columns to be used for training/prediction
X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_inputs[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

# Create and train the model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, train_targets)

# Generate predictions and probabilities
train_preds = model.predict(X_train)
train_probs = model.predict_proba(X_train)
accuracy_score(train_targets, train_preds)

# Helper function to predict, compute accuracy & plot confustion matrix
def predict_and_plot(inputs, targets, name=''):
    preds = model.predict(inputs)
    accuracy = accuracy_score(targets, preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    cf = confusion_matrix(targets, preds, normalize='true')
    plt.figure()
    sns.heatmap(cf, annot=True)
    plt.xlabel('Prediction')
    plt.ylabel('Target')
    plt.title('{} Confusion Matrix'.format(name));    
    return preds

# Evaluate on validation and test set
val_preds = predict_and_plot(X_val, val_targets, 'Validation')
test_preds = predict_and_plot(X_test, test_targets, 'Test')

# Save the trained model & load it back
aussie_rain = {'model': model, 'imputer': imputer, 'scaler': scaler, 'encoder': encoder,
               'input_cols': input_cols, 'target_col': target_col, 'numeric_cols': numeric_cols,
               'categorical_cols': categorical_cols, 'encoded_cols': encoded_cols}
joblib.dump(aussie_rain, 'aussie_rain.joblib')
aussie_rain2 = joblib.load('aussie_rain.joblib')


# > **EXERCISE**: Try to explain each line of code in the above cell in your own words. Scroll back to relevant sections of the notebook if needed.

# ### Prediction on Single Inputs 

# In[ ]:


def predict_input(single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols])
    X_input = input_df[numeric_cols + encoded_cols]
    pred = model.predict(X_input)[0]
    prob = model.predict_proba(X_input)[0][list(model.classes_).index(pred)]
    return pred, prob

new_input = {'Date': '2021-06-19',
             'Location': 'Launceston',
             'MinTemp': 23.2,
             'MaxTemp': 33.2,
             'Rainfall': 10.2,
             'Evaporation': 4.2,
             'Sunshine': np.nan,
             'WindGustDir': 'NNW',
             'WindGustSpeed': 52.0,
             'WindDir9am': 'NW',
             'WindDir3pm': 'NNE',
             'WindSpeed9am': 13.0,
             'WindSpeed3pm': 20.0,
             'Humidity9am': 89.0,
             'Humidity3pm': 58.0,
             'Pressure9am': 1004.8,
             'Pressure3pm': 1001.5,
             'Cloud9am': 8.0,
             'Cloud3pm': 5.0,
             'Temp9am': 25.7,
             'Temp3pm': 33.0,
             'RainToday': 'Yes'}

predict_input(new_input)


# Let's save our work using Jovian.

# ## Summary and References
# 
# Logistic regression is a commonly used technique for solving binary classification problems. In a logistic regression model: 
# 
# - we take linear combination (or weighted sum of the input features) 
# - we apply the sigmoid function to the result to obtain a number between 0 and 1
# - this number represents the probability of the input being classified as "Yes"
# - instead of RMSE, the cross entropy loss function is used to evaluate the results
# 
# 
# Here's a visual summary of how a logistic regression model is structured ([source](http://datahacker.rs/005-pytorch-logistic-regression-in-pytorch/)):
# 
# 
# <img src="https://i.imgur.com/YMaMo5D.png" width="480">
# 
# 
# To train a logistic regression model, we can use the `LogisticRegression` class from Scikit-learn. We covered the following topics in this tutorial:
# 
# - Downloading a real-world dataset from Kaggle
# - Exploratory data analysis and visualization
# - Splitting a dataset into training, validation & test sets
# - Filling/imputing missing values in numeric columns
# - Scaling numeric features to a $(0,1)$ range
# - Encoding categorical columns as one-hot vectors
# - Training a logistic regression model using Scikit-learn
# - Evaluating a model using a validation set and test set
# 
# 
# 
# 
# Check out the following resources to learn more:
# 
# * https://www.youtube.com/watch?v=-la3q9d7AKQ&list=PLNeKWBMsAzboR8vvhnlanxCNr2V7ITuxy&index=1
# * https://www.kaggle.com/prashant111/extensive-analysis-eda-fe-modelling
# * https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction#Baseline
# * https://jovian.ai/aakashns/03-logistic-regression
# 
# 
# 
# 

# In[ ]:




