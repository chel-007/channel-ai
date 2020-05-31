---
layout: post
author: Chel
title: Introduction to Linear Regression - Creating a Simple Learning Predictor
img-src: ../assets\images\Blog\coming-soon.jpg
---

So now that you know a little bit about Machine Learning. It becomes natural for you to understand how you can use it to build models that are capable of making decisions at minimal level. The way we achieve this is the use of Machine Learning Algorithms that were developed several years back by experts in the field.

The way these algoritm are capable of learning from real-world data to be able to make predictions is mostly because the mathematical computations on which they are built support or are proven to learn or map data to specific labels through probability and with experience.

In this article, you would be introduced to one of the simplest and popular ML Algorithm called <strong>Linear Regression</strong>. It is recommended that you read our tutorial on <em><a class="blog-links" href="">Vectorization with Matlab</a></em> if you haven't already as we would be using vectorization to implement our LR model for faster and easier bug debugging.

Linear Regression is a ML algorithm that is used for predictions of "continuous /non-classified data". To better understand it, let's use a very popular example. Say, you want to predict the prices of particular houses,height of students in a class, weight and so on. This examples define continuous data, that is, they stand alone and are not grouped into classes. This is in contrast to a second learning algorithm called Logistic Regression which we would be looking at in the next post. Before we proceed to the implementation of Linear Regression, i want to give a general idea or intuition about what an ML model is trying to achieve and how it works.

I learnt Linear Regression from the Stanford University <em><a class="blog-links" href="">Machine Learning course on Coursera</a></em> taught by Professor Andrew Ng. I feel like i could not have gotten a better understanding if i had learnt it elsewhere, Andrew always wanted to give the students an intuition about any new algorithm we were taking on and i loved it. It made me understand exactly how that algo was different from the previous one even before we started learning the maths behind it. I should say, that's one of the best course or teaching i have ever received. Now i have this understanding of the different algoritms that even if i were to forget the correct code to implement them i still have a very good grasp of what they are doing and i think that's very important.


Now i would give you guys that intuition at the start of every algo you would be learning.

You have some datasets in continuous values(because we are doing LinearRegrssion) and you want to go about training a model to the point that it achieves a preferred level of accuracy that you consider good enough for the use you want. Yh yh, we want a model that can do that for us but <b>how</b> does it come to achieve this?. In Machine Learning we have two main types of model learning; supervised and unsupervised. To get more insights on them, read our full article on <em><a class="blog-links" href=""><b>Machine Learning</b></a></em>. In Linear Regression we are carrying out the supervised part of learning, think of it as supervised beacuse we are providing some labels or correct values that we want our model to learn from. It goes exactly like this: we have a data set we a number of examples, each of this examples has two values in a row (x and y). The x denotes the value that is fed into the model and y denoting the output of the model. These x's and y's are both fed to the model during the training step/phase, therefore the name supervised. 
When we begin to do the math of it, i would explain i details what these terms mean and you should hopefully understand it better.

We have some specific steps of training that we would take iteratively or repeatedly, the main purpose of this is to reduce the errors generated at each trainng step or loop. Its almost as easy as that. Now we come to the algoritms involoved that helps us achieve that.

THE THREE MAIN CODE IMPLEMENTATION WHEN TRAINING A LINEAR REGRESSION ALGORITHM:

1. Prediction function
2. Cost Function
3. Gradient Descent

Let's look at each of them closely. We know we want an accurate "predictor", and for that to be, three things has to happen: Our model could start by randomly guessing or well thinking at least it's guessing correctly but the model would definitely be wrong because initially it hasn't mapped a function to the data, now we use the cost function to calculate the errors gotten on all our training examples of that step in the loop,that is. the errors when our predictor tried to guess what the output y's of each x's were. Finally, the Gradient Descent algo tries to fit our model better by reducing the amount of those errors until it converges to a point that the errors are minimal or can't be reduced further. Now i would define all the most of the terms used in Linear Regression so you can understand better what each is doing.

1. Dataset: Dataset is a collection of data, usually data fron real world affairs that can be used to train a model. It can be in either <em><a class="blog-links" href="">structured</a></em> or <em><a class="blog-links" href="">unstructured</a></em> form. The stuctured type of data can be divided into <em><a class="blog-links" href="">supervised</a></em> or <em><a class="blog-links" href="">unsupervised</a></em> datasets. Datasets sizes ranges from hundreds of trainig examples to as much as millions. Data is usually the first step of model training referred informally as "collecting data".

2. Algorithm: A set of mathematical or statistical calculations that is used to compute for numbers, they are proven to give us our answers if implemented in the correct order. Algorithms are found in different fields of study, each constucted to be abstracted will also supporting some higher-level logic for the purpose it's used.

3. Training Examples: Usually found in the dataset, training examples in Linear Regression consists of some x's and y's which we want our model to learn from considering they are the correct form of matching or mapping that we want. Think of the input x as any subject that can be used to 'determine' an effect or some other number. And y being that number that was the result of exactly what <b>x</b> is. So let's just say they are dependable of themselves.They are denoted by <b>m</b>

4. Cost Function: A function in Machine Learning that is always given the task of finding errors after one iteration of learning or after our "predictor" as tried to guess the output of all x's. Different ML algorithms would come with cost functions that are constructed for exactly how they are used or what concept they support, that in most cases would not work for a different Ml Algorithm. We would call the output y the model is trying to predict <b>y hat</b>. Consequently, this cost-function is the difference between yhat and y matched be some parameter theta. They are denoted by capital <b>J</b>

5. Parameters: In Linear Regression, we are trying to predict our y's given x's. How we achieve this is that we use parameters called theta to map the function for us. These theta are practically the ones that gives us our errors and by changing them, we can get smaller errors and so on. They are what the model is built upon. In Linear Regression, parameters theta multiplied by our input data x gives us our prediction of ouput y. So you can see how important they are, we use gradient descent to change or reduce the errors calculated by cost function and by changing those errors our parameters theta are defining the predictions a bit more correctly, that is, they are getting better.

6. Gradient Descent: Gradient Deascent is a powerful algorithm that is used in Linear Regression to update theta to a value that gives less errors. It is an iterative function that keeps updating theta in every cycle of the loop until theta converges or gets to the point that it's reduction if any becomes insignificant.

So now you know the three main steps involved in Linear Regression and hopefully the basic definitions of those terms gave you a better idea of how we train a Predictor or model. Now we can go ahead to implement all these in code. For this tutorial, I would get a dataset of one feature X and output Y from <em><a class="blog-links" href="https://www.openml.org/search?type=data">Open ML</a></em>,a platform where ML engineers and data scientists can collaborate on projects or share datasets. I will be implementing and testing the code written here on <em><a class="blog-links" href="">MATLAB</a></em>, a platform for carrying out mathematical computations and linear algebra. It comes with a free-trial, but to use for an extended period of time, a fee has to be paid. If you want to follow along, you can either use MATLAB or OCTAVE. To set up octave, read our article on <em><a class="blog-links" href="">how to set up octave on windows</a></em>.

So I downloaded a dataset from open ML to my PC, I can then login to my MATLAB acct, upload the data, and start building a Linear Regression predictor. Let's Begin

AIM OF PREDICTOR

In this tutorial we would be building a simple model that would use a number of input features x to predict the 'price' a building will sell for.

DATASET

<pre><code class="nohighlight">This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.

It contains 19 house features plus the price and the id columns, along with 21613 observations. It's a great dataset for evaluating simple regression models.</code></pre>

You can download this dataset from OpenML <em><a class="blog-links" href="https://www.openml.org/d/42092">here</a></em>. As stated it is a multi-varaint dataset or simply, it has multiple features in total 19, and 21613 training examples. For this tutorial, we would be using 5 features and 100 training examples.To implement for a uni-variant model, you can follow the whole process but instead load just one feature into the x variable. You would learn about it all soon.


FEATURES

The 5 features we would be giving our variable x are <strong>number of bedrooms</strong>, <strong>no of bathrooms</strong>, <strong>sqft of living area</strong>, <strong>floors</strong> and <strong>year built</strong>. There a a lot more features but the reason i chose those 5 is because they seem like the factors that play the largest role of influence on the price a customer would like to pay for a house. Also including similar features like <strong>sqft of basement , sqft of upperlevels</strong> and so on would end up making the features redundant (repetitive or duplicating). Take a look at the full features in our dataset.

<img class="img-fluid" src="../assets\images\Blog\Blog-img\lr-feature-list.png">

To set off,we would be using the list as a guide to know the order of the steps it takes to build our model.You can always refer back to this, as a reminder of what steps you should be implementing at what time.

1. Plot the data
2. Feature Normalization
3. Compute for the Cost Function
4. Gradient Descent

There is a step included that we have not yet covered, let's do that now. 

Feature Normalization is a way to neutralize the wide range of values we might have in our dataset. It simply reduces how far apart how data is in their real values in numbers. For example, in  our house_sales dataset we can observe that the difference between the year_built and no_of_bedrooms features is so wide apart and can cause irregularities for our model to learn accurately in lesser time. The way to combat this is simply to normalize our data. In Machine Learning this is referred to as "Feature Normalization". Now let's begin with the first step on the list.

Plotting data is a good way to understand our data, it can be done at different steps when implementing ML algorithm, apart from the purpose of visualization, it also helps us for debugging just by looking at the curves and shapes our data takes.You would learn more about this in future lessons. To plot the data onto a graph, we use a set of built-in functions from MATLAB (same applies in Octave). Whenever i use some new built-in function in our code I would give a brief explanation of what it does, for this reson, anyline starting with a "%" signifies an explanation of the code on the next line. The "%" sign is the official way used to write comments in Matlab.


<pre><code class="matlab">
% Load Data into Matlab Workspace
data = load('dataset_');
a = data(:, 4,5,6,8,15);
</code></pre>