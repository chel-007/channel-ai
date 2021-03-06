---
layout: post
author: Chel
title : Beginners Guide to Feature Selection and Categorical Embeddings with Project on Unclean Structured Data
img-src: ../assets\images\Blog\feature-select
img-alt: feature selection categorical embeddings unstructured data
description: Most of the times, when attempting to visualize and build deep learning models using real-world data, you usually never get a dataset fiiting your requirement closely. In this case, it becomes necessary to apply data cleaning methods, which is the subject of this tutorial.
---


Have you ever been faced with such a dataset so unclean and irregular, that you feel terrified? 😯. If your answer is NO, you have either been practicing deep learning for over 3 years or you rarely ever practice at all? But jokes aside, we deep learners frequently get faced with rather unclean data that needs tedious amounts of processing to be usable. Several of these problems which we are gonna tackle in this tutorial by solving a machine learning problem in a beginner-friendly way. let's get started!


In this Tutorial(We Cover the Foll):
* Real World Structured Data
* Example Project
* Dataset Visualization with Sns and Pandas
* Tackling Nan Values
* Performing Categorical Embeddings
* Data Imputation Methods
* How to select priority features (Feature Selection)
* Fitting Model
* Feedback and Summary

**NOTE: If at any point in time, you feel confused, make sure to check my Colab notebook and follow along with it so you can see the full transfer of data between data frames**
<a href="https://colab.research.google.com/drive/1ZqMuTzZpzaTnubjXlfZ49MuAisfXGPpx#scrollTo=4PMMzA96mT6A">Full Notebook Summary</a>

### Structured Data in the Real World
Structured Data is Data that can be tabulated or visualized in a table format. In machine learning, this term refers to numerical Data also known as Continous Values. Structured Data is logically the opposite of unstructured data, which refers to data that can not be technically tabularized or which has an irregular format, examples are, Images, text and videos data


> Structured data conforms to a tabular format with a relationship between the different rows and columns. Common examples of structured data are Excel files or SQL databases.
> — <cite>Big Data Framework[^1]</cite>

In the real world, we gather a lot of structured data, which follows the laws of the real world. When they are collected, no consideration is given to how they can be used for machine learning and collecting insights. An Example: Imagine, a law firm kept the records on all their customers in the year 2019, with information on say, how long the company's representatives talked on the phone with customers, how much money customers invested for insurance, etc. In a record as this, there are natural data that are not readily convertible for working with. In this case, employing machine learning cleaning methods is the only viable open. Let's dive into the subject data for this tutorial.

### Example Project
The dataset for this tutorial contains two files:
* train.csv: 6500 X 20
* test.csv: 3500 X 19

In other words, we have a training data frame with 6500 rows and 20 columns and a testing/evaluating data frame with 3500 rows and 19 columns.

###### The task
You work for a company that sells sculptures that are acquired from various artists around the world. Your task is to predict the cost required to ship these sculptures to customers based on the information provided in the dataset.
The data frames contain several columns which depict the features we are working with, and we need to build a model that predicts the last column in the test set(Cost of Sculptures). Let's Visualize the dataset to see what it looks like.


#### Dataset Visualization with Sns and Pandas

To visualize the dataset, we should load it up, import the necessary packages & modules. I'm working on Colab for this tutorial and the dataset is stored in my drive. To follow along with this guide , download dataset <a href="https://drive.google.com/file/d/12q8kY1MYKpyTMPvk8aNDa-TindyAmsGg/view?usp=sharing">here</a>. Upload the zip file to drive
* The first thing we need to do is to mount Drive:
{% highlight python linenos %}
from google.colab import drive
drive.mount('/content/drive')
{% endhighlight %}


* Import necessary packages which we would use throughout the project
{% highlight python linenos %}
import os
import random
import tensorflow as tf
import numpy as np
import pandas as pd 
# Use seaborn for pairplot
import matplotlib.pyplot as plt
import seaborn as sns
{% endhighlight %}


* Next, let's import the Zip module and extract the dataset. On this occasion, after having learned how using the alias name 'zip' when calling ZipFile class result in errors for later use, I decided to use a different name i.e grab.

{% highlight python linenos %}
from zipfile import ZipFile
file_name = '/path/to/dataset'
with ZipFile(file_name, 'r') as grab:
grab.extractall('/path/')
print('Done')
{% endhighlight %}


* Now with the dataset loaded, we can run some visualizations using pandas and sns. First, let's take a look at the columns in the train.csv file using pandas to read , draw the head, then quickly check for Nan Values.
{% highlight python linenos %}
raw_dataset = pd.read_csv('/path/to/extracted/csv')
dataset = raw_dataset.copy()
dataset.head()
# Checking for nan values
print(datasetisnull().sum())
{% endhighlight %}

In the image below, we can see the total of 20 columns and their features; **Customer ID, Artist Name, Artist Reputation, Width, Height, Width, Material, Price of Sculpture, Base Shipping Price, International, Express Shipment, Installation Included, Transport, Fragile, Customer Information, Remote Location, Scheduled Date, Delivery Date, Customer Location**. We also have a lot of Nan Values amongst our data, this would be tackled after dropping irrelevant columns with string values and over-tedious formats like a date.

<img src="">


### Tackling Nan Values & Irrelevant features

Some particular columns contain string values and other formats that are irrelevant in predicting the cost of the artwork.

For this next step, we have two options to consider, should we:
* Start Categorical Embeddings
* Or do we first drop columns we wouldn't need.

Let's approach this somewhat logically: If we start with the first option, we are likely going to meet with columns that can't be(or rather, too tedious) to embed. But if we drop them first, we can exclude those absurdly irrelevant features before moving on embedding the relevant portion of columns.

Before Dropping Columns in the training and test set, let's first pop-out our target column (Cost) and save it. Later we can reference it for fitting our model.

{% highlight python linenos %}
X, y = raw_dataset.drop('Cost', axis=1), raw_dataset.iloc[:,-1:]
y.head()
{% endhighlight %}

In the above code block, we separated our cost from the main train data frame and stored it in a new df. On drawing the head though, we can see something strange at work!. The cost consists of both negative and positive values, this won't do, we want **Only** positive values. As I said earlier real-world data never comes the way you expect. This could just be a wrong entry by a tired cashier or a more acceptable answer perhaps would be that (For all Art Sculptures that were delivered with defects, the company holds the loss?). Anyway, we can easily correct this using the absolute function in pandas. `The absolute of any number is = the **positive of that number**`

{% highlight python linenos %}
train_Y = y['Cost'].abs()
train_Y.head()
{% endhighlight %}

With the above code, our target variable now contains only positive values. Now that's taken care of, we can proceed to drop special irrelevant columns from our training set

{% highlight python linenos %}
import pandas as pd
import numpy as np  
X = raw_dataset.copy()
train = X.drop(['Cost','Customer Id','Artist Name','Delivery Date','Scheduled Date','Customer Location'], axis=1)
{% endhighlight %}


Earlier, I checked our columns for nana values. In total it returned about 4000 Nan Values. But in every column we see that nan values are occurring more in some especially (Material, Transport, Remote_Location, Width, Height, etc). For this step, we want to replace nan values in *First* Columns with more than two classes. Material, Remote_Location, and Transport fit the description, the reason is this, if we embed them we stand a chance of losing the relative relationship in our features. To do this, we are gonna create a simple function that fills Nan values with the highest occurring category in the column. e.g In the Material column, we have 7 classes (Brass, Stone, Aluminium, Bronze, Clay, ). Our function loops through the rows and replaces Nan Values with the highest repeating class.

{% highlight python linenos %}
#Function to replace NAN values with mode value
def replace_nan_most_freq(DataFrame,ColName):
most_frequent_category=DataFrame[ColName].mode()[0]
replace nan values with most occured category
DataFrame[ColName +"-Imputed"] = DataFrame[ColName]
DataFrame[ColName + "-Imputed"].fillna(most_frequent_category,inplace=True)
#Call function to impute most occured category
for Columns in ['Material', 'Remote Location', 'Transport']:
replace_nan_most_freq(train,Columns)
replace_nan_most_freq(test,Columns)
# Display imputed result
train[['Material','Material-Imputed','Remote Location','Remote Location-Imputed','Transport','Transport-Imputed']].head(10)
test[['Material','Material-Imputed','Remote Location','Remote Location-Imputed','Transport','Transport-Imputed]].head(10)
#Drop actual columns
train = train.drop(['Material', 'Remote Location','Transport'], axis = 1)
test = test.drop(['Material', 'Remote Location','Transport'], axis = 1)
{% endhighlight %}



With that, if we plot the train and test df, we get the following:
As you can see Material has been replaced by Material_Imputed, likewise with Transport and Remote Location.;
<img src="">

### Categorical Embeddings


In categorical Embedding, we want to check for columns with a data type of `object` and label encode them using scikit's learn Label Encoder.

{% highlight python linenos %}
# Get list of categorical variables
s = (train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
{% endhighlight %}


{% highlight python linenos %}
from sklearn.preprocessing import LabelEncoder

# Make copy to avoid changing original data 
label_X_train = train.copy()
label_X_test = test.copy()

# Apply label encoder to each column with categorical data
label_encoder = LabelEncoder()
for col in object_cols:
    label_X_train[col] = label_encoder.fit_transform(train[col])
    label_X_test[col] = label_encoder.fit_transform(test[col])
{% endhighlight %}


With that, plotting the head gives us this: 
<img src="">

All category columns have been lav\bel encoded to numerical values. The last step is to fill up the remaining columns that still contain Nan Values like Width, Height, Weight. Because, if you can remember, we only replaced the category nan values above. The code below uses sklearn's Simple Impueter method.

{% highlight python linenos %}
from sklearn.impute import SimpleImputer
# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(label_X_train))
imputed_X_test = pd.DataFrame(my_imputer.fit_transform(label_X_test))
# Imputation removed column names; put them back
imputed_X_train.columns = train.columns
imputed_X_test.columns = test.columns
{% endhighlight %}


Now if we check for Nan values, it returns zero across all columns.


### Feature Selection

With that taken care of, we can now start doing some feature selection and deciding which columns are of no use to us. A lot of logical thinking is important as this is a real-world problem and real-world insight is required.

**NOTE: Any column we decide to drop must be reflected in the test set. Likewise, any preprocessing step we take. As this data(test set) is what we would be making predictions on, and inconsistent columns would result in errors.**

## How to select Priority Features (Methods you can use)
When faced with a feature selection problem in deep learning and machine learning, there are several methods you can apply to arrive at better features during training, therefore, better model accuracy. 

We have *now* 19 columns in our training set after dropping Cost.

#### Description of variables in the above file

*	**Customer ID**: A set of unique values associated with every customer (This adds absolutely nothing of value)

*	**Artist Name**: The name of the Artist who created the artwork
*	**Artist Reputation**: A float value important for understanding how expensive an art would be
*	**Weight**: How heavy an Artwork is, but as humans, we know that heavier artwork doesn't necessarily mean more expensive. But still a great judge of worth.
*	**Width**: Width of Artwork
*	**Height**: height of the artwork
*	**Material**: A very important feature
*	**Base Shipping Price**: The original cost for shipping artwork from one location to another. It varies (Important)
*	**International**: Geographic location of a customer. Across borders means extra customs cost (Important)
*	**Express Shipment**: Value depicting if a Customer requested for express shipment (Important)
*	**Installation Included**: Should artwork be installed along with delivery? (Important)
*	**Fragile**: How fragile the artwork is. Just like weight, more fragile doesn't mean more expensive. It's a shaky yet important feature (Stable)
*	**Transport**: What means of transportation is used to deliver Artwork. Airplanes generally cost more (Important)
*	**Remote Location**: What kind of Environment Customer resides. The lesser accessible the more cost it takes?. This feature is a little bit unspecific because it could easily be the other way round. A poorer customer is less likely to pay more. (Stable)
*	**Customer Information**: How financially stable is the purchaser. More means more likely to give tips, request installations, quicker delivery, etc. (Important)
*	**Customer Location**: Where a customer resides. A bunch of specific locations that would be too tedious to embed. Also, Remote Location gathers similar Information (Redundant)
*	**Delivery Date**: Date purchaser ordered for ar work to be delivered. A good **BUT** redundant feature. features like express shipment and installations are better judges of how quickly customer wants artwork (Redundant)
*	**Scheduled Date**: Same as Delivery Date (Redundant)


The above can be achieved with little logical deductions and insights, but as you can see, there are still some uncertainties. By using methods, tests, and libraries we have a better ground to decide.

1. <h3>Univariate Selection</h3>
Statistical tests can be used to select those features that have the strongest relationship with the output variable.
The scikit-learn library provides the SelectKBest class that can be used with a suite of different statistical tests to select a specific number of features.
The example below uses the f_classif statistical test for positive numerical features to select 10 of the best features from the Art Exhibition Dataset. Some of the feature columns must be dropped before you can use this method: Those that are string values e.g Customer Id and Artist Name.

{% highlight python linenos %}
#apply SelectKBest class to extract top 10 best features
from sklearn.feature_selection import SelectKBest, f_classif
bestfeatures = SelectKBest(score_func=f_classif, k=10)
fit = bestfeatures.fit(x_final,train_Y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(x_final.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features
{% endhighlight %}


The top ten best features are displayed below helping us get rid of uncertainties like the `Fragile` column.

<img src="">

2. <h3>Feature Importance</h3>
You can get the feature importance of each feature of your dataset by using the feature importance property of the model.
Feature importance gives you a score for each feature of your data, the higher the score, the more important or relevant is the feature towards your output variable.
Feature importance is an inbuilt class that comes with Tree-Based Classifiers, we will be using Extra Tree Regressor for extracting the top 10 features for the dataset.

{% highlight python linenos %}
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
model = ExtraTreesRegressor()
model.fit(x_final,train_Y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=x_final.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
{% endhighlight %}


<img src="">


Again, we see that some certain feature columns are among the top 10 using this selection method. We now know those very important features we should target when we limit the training features to say, 10.


3. <h3>Correlation Matrix with Heatmap</h3>
Correlation states how the features are related to each other or the target variable.
Correlation can be positive (increase in one value of feature increases the value of the target variable) or negative (increase in one value of feature decreases the value of the target variable)
Heatmap makes it easy to identify which features are most related to the target variable, we will plot a heatmap of correlated features using the seaborn library.

{% highlight python linenos %}
#get correlations of each features in dataset
corrmat = raw_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(raw_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")
{% endhighlight %}

<img src="">


Now, we are talking. Let's make a round-up of the 11 features we want to use for training, by selecting those occurring the most after the feature selection methods above.


### Ten Best Features to Use
1. Price of Sculpture
2. weight
3. Artist reputation
4. Base Shipping Price
5. Width
6. Height
7. Express Shipment
8. International
9. Transport_Imputed
10. material_Imputed
11. Customer Information.



### Fitting the Model

We would quickly drop the remaining columns while leaving the above, and Finally, proceed with scaling our data & fitting our model with sklearn StandardScaler and Random Forest Regressor respectively.

{% highlight python linenos %}
x_train = imputed_X_train.drop(['Fragile','Remote Location','Installation Included'], axis=1)
x_test = imputed_X_test.drop(['Fragile','Remote Location','Installation Included'], axis=1)
{% endhighlight %}



{% highlight python linenos %}
import sklearn
scaler = sklearn.preprocessing.StandardScaler()

x_train = pd.DataFrame(scaler.fit_transform(imputed_X_train), columns=imputed_X_train.columns)
x_test = pd.DataFrame(scaler.fit_transform(imputed_X_test), columns=imputed_X_test.columns)Location','Installation Included'], axis=1)
{% endhighlight %}



{% highlight python linenos %}
#Import Random Forest Model
from sklearn.ensemble import RandomForestRegressor

#Create a Gaussian Classifier
rgf=RandomForestRegressor(n_estimators=100, random_state=42)

#Train the model using the training sets y_pred=clf.predict(X_test)
rgf.fit(x_train,train_Y)
{% endhighlight %}


Last But not least, we need to evaluate our model on the test set, prepare it and arrange it in a new data frame with two columns Customer Id and Cost.


And now this is how the final model looks like. You can download the .csv file <a href="">here</a>.


### Feedback and Summary
We have had fun with this project, cleaning, visualizing, dropping, performing feature engineering, and learning how to use the sklearn library for Machine Learning work. It's a comprehensive guide where I decided to tackle problems to show you how you can solve similar problems. Naturally, you have questions to ask, and I'm only happy to see you ask them. The comments box below is available for your chats. Thanks for reading this article, I hoped it has achieved its purpose of guiding you through the concept of feature selection engineering in deep learning, Chel.



[^1]: The above quote is a definition extracted from Big Data Framework's [article](https://www.bigdataframework.org/data-types-structured-vs-unstructured-data/) written on January 9th, 2019.





