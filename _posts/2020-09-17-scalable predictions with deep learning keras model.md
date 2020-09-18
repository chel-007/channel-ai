---
layout: post
author: Chel
title: How to make Scalale Predictions with a Deep Learning Model Built with Keras
seo: deep learning python dl python deep learning what is deep learning deep learning deployment tensorflow deep learner 
description: Buiding Models is exciting, putting them to use can be very brain-tasking. For people who have mastered model deployment, usually this involves just a few tweaks to the code to make it work for a new problem and situation. For beginners like you and me, the task of scaling models for prediction for use in different environment requires a lot of information gathering, software set-up and a little bit of technical know-how. In this article, my aim is to simplify this process for you and show you the methods and steps you can immediately put to use in your projects.
img-src: ../assets\images\Blog\dl_predictions
---

Buiding Models is exciting, putting them to use can be very brain-tasking. For people who have mastered model deployment, usually this involves just a few tweaks to the code to make it work for a new problem and situation. For beginners like you and me, the task of scaling models for prediction for use in different environment requires a lot of information gathering, software set-up and a little bit of technical know-how. In this article, my aim is to simplify this process for you and show you the methods and steps you can immediately put to use in your projects.

These are the steps guiding us through this article:

<h3>Overview</h3>
<ul>
  <li>Introduction</li>
  <li>Buidling Deep Learning Models in Keras</li>
  <li>Making Predictions: Methods to Utilize a Model for Predictions</li>
  <li>Summary</li>
</ul>  


<h3>Introduction</h3>
As a beginner, usually you spend most of your time building models by leveraging blog posts tutorials, <a href="http://kaggle.com/">kaggle data-science</a> community or follow-up from courses you take. At least this was what most of my time has been spent doing.

The beauty about model deploment and prediction is that you  can see how your modelperforms in a real-world situation which is an important step that develops a very useful skill every Machine Learning Enginner should possess. No-one wants an enginner that cannot solve problems with their models because it remains sitting on google colab or on thier machine.

There is one quality every smart ML Enginner or Data-Scientist possesses and that is "the ability to know the best procedure for every unique task or problem". I say this because, in deep learning no two project is the exact same thing, there are always a few tweaks that must be made to fit a certain unique problem. In model deployment, this same fact holds true and later in the article, an example would be given that portrays how two problems has different routes that ensures the best results. That said, it is necessary to understand the various methods of how you can make scalable predictions from your deep learning models.

<h3>Buidling Deep Learning Models in Keras</h3>
It is assumed that you have some practise building DL models, at least you have been doing this for a month upwards. This way, and at this point you would know the process of building models with CNNs, RNNs, ML Algorithms and extended libraries and also saving models in your preferred format. Just in case this is not the case for you or perhaps you just need some brush up summary, I would dive briefly explaning how to build deep learning models.

In this article, I refer to DL models built with Tensorflow, on top of Keras. I use this for two main reasons. One is that, I have the best practise uilding models with Tensorflow and Two, I believe that keras has very easy steps to build models and there exist many methods for deployment and scalable predictions.

Keras is a high-end library for building ML and DL models. It is built on top of Tensorflow and utilizes it's many tools, functions and libraries for data loading,visualization,pre-processing and the wholw process involved in model building.

Most ML and DL task can be carried out in Keras like Computer Vision, Natural Language Processing, Style Transfer, Transfer Learning, Gans, pretty much any DL task can be carried out using keras. If you want some folllow up tutorials for Dl in keras, I suggest you visit Tensorflow oficial website <a href="https://www.tensorflow.org/tutorials/quickstart/beginner">here</a>.

<h3>Making Predictions: Methods to Utilize a Model for Prediction</h3>
Making predictions with models is a successding step of model deployment. In a way, it can be used interchangably. "I want to deploy my model", for what?,"to make predictions".

This section holds the main content of this article, where I would give you the methods, merits/demerits and all the necessary steps you would need to take to apply any of the <b>scalale predictions and deployments routes</b> below. Before we get to that, I want to take a moment to answer some questions beginners usually ask. If you don't consider yourself a beginner, go ahead and move to the next section.

<h5>Questions beginners might ask about Deployment</h5>

"Is model deployment necessary to a beginner in ML and DL" - No, this should not be something that you spend time on trying to implement because this is going to waste a lot of your time that could be better invested in practising with simple projects

"Can I deploy a model that doesn't solve a very worthy real-world problem"- I absolutely believe that you can do this if you have the time and resources for it because it serves as a learning process and if you start building 
models solving simple problems, you would soon start building advanced ones if you persist in it.

"What is the main point of model deployment"- The main point is to provide the function of your deep learning model to a wide audience of people and that it should be easily accessile from various devices.

"As a beginner, what would be the point of deploying a model aside from feeling good with myself"- Models are meant to solve problems, so yhh if you can build a model channeled to solve or simplify a particular problem, you can distribute this and have people use it as a paid or free service. This is very logical, and what most business do just in advanced applications.

These were questions that troubled me when I got to know what deployment is all about. I hope that they could be of some help to you. 
Below I am going to take you through the process of model deployment and how you can go about it. Using this guide, you should get through in a day what would naturally have taken upwards of a week.

<h3>Methods to Deploy a model for Scalale Prediction</h3>

<ol>
	<li>TFX(Tensorflow Extended): A TFX pipeline is a sequence of components that implement an ML pipeline which is specifically designed for scalable, high-performance machine learning tasks. That includes modeling, training, serving inference, and managing deployments to online, native mobile, and JavaScript targets. To build this pipeline, a few libraries are tasked to take care of different steps in building the scalable ML Pipeline.They are: <ul style="list-style-type:circle;">
		<li>TensorFlow Data Validation for validating, analysing and monitoring ML data at scale. It helps to maintain the health of the ML pipeline</li>
		<li>TensorFlow Transform for preprocessing data into suitable formats. It involves tokenizing and numerical operations such as normalization.This is what i was referring to at the beginning of the post that every model must be prepared for predictions in a different format</li>
		<li>TensorFlow Model Analysis for computing visualization and evaluation metrics for models. Before deploment it is necessary to evaluate the quality of the model to ensure it meets desired threshold.</li>
		<li>TensorFlow Serving to support model versioning and for model updates with a rollback option and multiple models for experimentation via A/B testing, while ensuring that concurrent models achieve high throughput on hardware. What Tensorflow serving does in simple words is that it handles updates to your model, rollback options and pushing of multiple models, in a way, think of it as having similar functions to github.</li></ul>

For beginners, your main area of focus should be <b>tensorflow serving</b>, all other libraries mentioned is secondary and for more professional projects. In the next few lines, I will give instructions on how to set up a TFX for serving models:

* The main software you need for this is Docker, you can install it from this url <a href="">Install Docker</a>
* You also need to install Tensorflow 2.0 and Keras from their official website
* Finally, you must install tensorflow model server by pulling a docker image. This must be done on a 64-bit machine.
These are the three software you need to set up locally. Refer to this article on how to serve a classification model built tensorflow using Tensorflow serving, this should be a guide you can follow to apply it in your own projects <a href="https://www.tensorflow.org/tfx/tutorials/serving/rest_simple">Train and serve a Tensorflow model with Tensorflow Serving</a>

<h3>Merits & Demerits of Tensorflow Serving as a Deployment method</h3>
<ul style="list-style-type: square;">
	<li>It has a very robust functions that allows for many custom configurations</li>
	<li>It requires technical know-how to set it up and might look tough to first-timers</li>
	<li>It comes with tools for visualization, normalization , preprocessing, evaluation, analysis of data. This means that developers can carry out every necessary step with just TFX</li>
	<li>It makes it very easy to update and rollback models versions.</li>
</ul>
</li>

<br>
<li>GCP AI Platform: AI Platform makes it easy for developers, data scientists, and data engineers to streamline their ML workflows. AI Platform helps all users take their projects from ideation to deployment seamlessly. There are four main steps involved:

* <b>Preparation</b> to store your datasets with BigQuery, then use the built-in Data Labeling Service to label your training data by applying classification, object detection, and entity extraction, etc
* <b>Build</b> for building your model with GCP Auto ML, a managed Jupyter Notebook service that provides fully configured environments for model development or in our case importing your complete tensorflow model.
* <b>Validation</b> to validate your model with AI Explanations that helps you understand your model's outputs, verify the model behavior, recognize bias in your models, and get ideas for ways to improve your model and your training data.
* <b>Deployment</b> to Deploy your models at scale and get predictions from them in the cloud with AI Platform Prediction that manages the infrastructure needed to run your model and makes it available for online and batch prediction requests.

The diagram below describes this process in details: <br>
<img class="img-fluid" src="https://cloud.google.com/images/ai-platform/ai_platform.svg"> <br>

From this brief introduction to AI Platform, we can see that we need to use only the deployment function of the lot. GCP provides the tools to prepare, train and validate but we can do all of these directly in keras when building a model. To get started with AI Platform for Deployment, visit <a href="https://cloud.google.com/ai-platform/docs/getting-started-keras">here</a>. The main advantage of AI Platform is that it is easier to use because most of the code is abstracted and handled b gcp under-the-hood. 
<b>Note:</b> GCP has it's pricing for all products, if you are just looking to experiment or start of somewhere you can use the 1 year bonus on gcp platform where you can use most products for free.
</li>

<br>
<li>Digital Ocean: The final option listed here is perhaps the easiest one to use. It is a very suitable options for beginners and small-medium sized enterprises. Yhh, you also don't have to read pages of documentation. All the work is done behind the hood, usually you just have to click a few options, write some code and that's it. Digital Ocean is by far the most popular cloud hosting service developers use, there are other options you can go for like Katerama, V etc.Digital Ocean offers a free 2 months trial period to test their services, this is a good way to learn how deployment and predictions work before been charged for it. To go about this process, you can refer to this article <a href="https://towardsdatascience.com/building-a-web-application-to-deploy-machine-learning-models-e224269c1331">Train and Deploy a Ml model built with Keras</a>.</li>
</ol>


That's it, with these guidelines, the whole process of deplotyment should be less intimidating to you. As a deep learner, this is a very important part of the process, and it is helpful to get accustomed to the process earler on.

Thanks for learning about deployment with me today, it is understandable if you have a few questions on your mind after taking in this huge pile of guidelines and directions, don't hesitate to leave me a comment and I'd be sure to get back to you. Chel_

