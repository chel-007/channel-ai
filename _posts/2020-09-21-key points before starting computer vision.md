---
layout: post
author: Chel
title: Starting your First Computer Vision Project? Here are 10 Things you Must Absolutely know
seo: deep learning computer vision deep learning deep learning deployment tensorflow deep learner keras guide for computer vision beginner computer vision
description: Can you imagine navigating through an unknown city without Google Maps?It is usually a tough and tiring ordeal, all paths seems to lead to nothing and at the end you even ask yourself why you took that path in the first place. That's often what the first Computer Vision project feels like. I can personally attest to this because I have been through it and now I know some real facts of what should and not be done.
img-src: ../assets\images\Blog\firstproject-cv
---

Can you imagine navigating through an unknown city without Google Maps?It is usually a tough and tiring ordeal, all paths seems to lead to nothing and at the end you even ask yourself why you took that path in the first place.

That's often what the first Computer Vision project feels like. I can personally attest to this because I have been through it and now I know some real facts of what should and not be done.

Learning the basics of CV and building a Computer Vision Model in tensorflow is great - but if you think that alone would land you a great job and carry you on to success, you'll be in for a big shock. In reality, it just isn't. For me, this reality hit home when I began building models on after the other, on colab solving projects on Kaggle, blog tutorials and official tensorflow website. There were other tons of things which were just as important, such as data collection, cleaning, exploration etc.

A few things I realized - problem solving skills, definite purpose, imagination, creativity are more helpful than mastering model building algorithms and tools. Trust me, this is way important than you might think.


In this article, I will be sharing 10 key points that I wish I knew when I started my Computer Vision journey, I hope this will help you out in your own CV journey too.



<h3>Stick to a Computer Vision Niche at First</h3>
As riduculous as it sounds, the firld of Computer Vision has niches. How can something so broad not have a niche?. If you wanna see for yourself just how, go to google.com and serach for the term "computer vision", there are almost 2 billion results for it. <br>
<img class="img-fluid" alt="computer vision search results image on google" width="50%" height="50%" src="/assets/images/Blog/Blog-img/cv-searchresults.webp">

I can't stress this enough, probably why it is placed as the first key point. When learning any new skill, one thing that you absolutely must try to avoid - Indecision. As a beginner, you are excited about this skill and want to try your hand in different paths thereby inviting time wasting which comes from Indecision. Trust me, I was once preparing for my First CV project so I know how it can be. What you want to do is stream line your thoughts to specific things only because your mind would be divided among different things to try. A simple question you can ask yourself: <b>What should you start with</b>, and my answer to that is "<i>start from the basics</i>". It significantly helps you on the long run. Almost every accomplished CV expert will give you the same advice even those that took a different path but later realized "basics are the roots and are very essential". When you start from the basics you gather the basic steps,libraries,algorithm(which might take time), but afterwards you gain a momentum that's double those that began straightaway pulling projects from github. A very good place to start that spells "basic" is Image Classification in Computer Vision. This article would get you started <a href="https://channelai.netlify.app/image">Building an Image Classification Model with Keras</a>.

<h4>How to decide between the niches</h4>
I mentioned niches in Computer Vision, you might be wondering what they are. Here I would give some examples on how Computer Vision is segmented today.
<b>Note:</b> Listed here are just a few from my knowledge that is, more experienced practitionerws would be able provide detailed examples and instances.
Some CV Niches are: 

* Classification projects like - Image Classification, Image Segmentation, Object Detection, Text Detection
* Synthesis projects like - Style Transfer, Image Colorization, GANs
* Real-time Prediction like - Image Captioning, Life face detection, Video recognition and analysis, Autonomous Vehicles.
It's obvious that they are listed in increasing other of complexity. Here you should start with classification problems which are generally the easiest, understanding them extensively before moving to others. This would make your confidence grow as you complete simple projects, feeling good with your new skill and ability would make you love CV and push on to bigger projects. Unlike when jumping between these niches which causes confusion, indecision and generally kills motivation. Remember that you are just getting started, not only do you have the right, it's necessary to start from the basic projects.


<h3>Knowledge of Computer Vision tools is great; Ability to break down data problems is priceless</h3>

<p style="text-align: center;"><i>Computer vision tools will come and go but the basics will stick forever</i></p>
Here is a very crucial step that beginners ignore without even knowing it. I have seen so many folks new to CV rush off to <b>master</b> specific tools and softwares with the hope that it would make them more effective and work smarter. That, is never the case. What is Effectiveness to begin?. Effectiveness is doing more in lesser time. Tools like Open CV,                         are designed to make you work faster. How do you put them to good use if you don't even know how to break a problem into actinable steps. Using this tools is great, but NOT at the expense of you learning problem solving skills and business intelligence. You can spend 2 whole months getting familair with an advanced CV tool. In reality, you don't even need it. Certainly not at beginner stage.
As an aspiring Computer Vision Enginner you have to create solutions by solving  real-world problems.

I will show you how you can go about developing the ability to break down data problems: You would need a lot of confidence at first, with time - you thrive in it. The best way to build these skills is by solving real world problems from communities like <a href="kaggle.com">kaggle</a>, <a href="zindi.com">zindi</a> and other deep learning hackathons you can find online.

Here I will give you a working cheatsheet you can use to learn, build and master problem-solving skills in Computer Vision (just follow the steps below):
* Open up Tensorflow Google Colab <a href="">here</a> and bookmark an empty notebook
* Creat an acct on kaggle or login in if you already have one <a href="">here</a>
* Go to your acct dashboard on kaggle and download your unique kaggle id to your computer. This is a json file
* Go to the projects section on kaggle, use the filter to narrow competitions down to "computer vision" & "beginner"
* When you find a project you like, open it up. Read the information of the project. Go to the data section and copy the url of this project. You will use this to download the project to google colab.
* Next copy these lines of code below. You would need to do two things - Change the url-of-the-project to the one you copied, - upload your own kaggle user id you previously downloaded when you are prompted for it.
<h5>code to set up kaggle project on colab</h5>
<pre><code class="css">
	
</code></pre>
* After completing the above steps the next step is for you to break down the problem, find resources and solve it. I recommend that before attempting a project from kaggle, you should have had experience using colab and have completed the <a href="">Image classification beginner project</a>. You can always come back to this article to begin, but do all these in the smallest possible time available to you. Don't creat time for procrastination.

By following these guidelines, you would no doubt develop problem-solving skills which is much needed to secure any CV job or becoming great at Computer Vision.

<h3>Model Deployment is key - Learn Software Enginnering</h3> 
One thing i see a lot of beginners do is avoid deployments of models with the excuse that it isn't important for them. What is Computer Vision without models we can put to use, honestly very irrelevant. What is the need of bothering about CV at all if you decide to not focus on the product - Deployment and Predictions. You build a working model on a jupyter notebook or colab, but it remains there waiting to be "read" and maybe referred to later when you need some code resources in it. That is not all the use case of a model. Models are meant to solve problems. Now though, let me tell you something else I have observed - Some folks choose to ignore deployment process because they are scared of the task "thinking" it just way too much work and also for more professional individuals. 

I would assure you now that - it really is not hard to deploy a CV model and make predictions with it, how do i know?, because I deployed a model on Image classification using tensorflow serving. So starting your first project in computer vision, keep model deployment in mind as an important step for success in CV. It is helpfulm to start it now, learn the ropes, so it won't be a total new experience when you need it most. This article will teach you everything you need to know about model deployment <a href=""></a>

<h3>Appreciate softwares with a simple learning curve</h3>
In any field of study two different softwares performing closely related tasks for users are usually cause for lots of argument comparation and reviews. In a good way, this is helpful to newcomers as it helps them make a choice of software that suits their needs closest and on ther other side of the coin, this brings about confusion and indecision. When I first statrted learning about CV, I was contemplating the ML framework to use for building models. I checked reviews, and honestly why it is always hard to make a singular choice and stand by it is because, Softwares have features they beat their competitor at and those they are lacking at. We want all the acctractive features to be present in one tools or software and usually this is never the case. As a beginner, a key guide to choosing the software you would employ in development process is to find the one with the easiet learning curve, not that with the best features - at first of course. You need a tool that you can assimilate in the shortest time possible, atool that doesn't take days to set up, in short a simple tool providing it allows for the basics of what you are searching for.

In Computer Vision and Deep Learning in general, we have a bunch of softwares. Concerning DL frameworks, people would argue for and against Tensorflow having the easiest interface and learning curve when compared to other frameworks like pytorch, scikit-learn, Theano, Apache Spark etc. But if I'm being absolutely earnest, Tensorflow is the framework every beginner should start with for these reasons:

* It has a large community of support in case you run into issues when developing
* It has a dedicated tutorial library for folks at any level of expertise right from "your first dl project(usally classifying hand-written digits". You don't need to look for Tensorflow projects online, you can start right from the website. And this is filled with all the important dl projects till date
* It has the easiest and fastest deployment methods. Tensorflow has a few tools you can use to push your complete model for real-world use. Examples are: Tensorflow Extended, Tensorflow.js, Tensorflow Lite. With these tools you can deploy to any platform you choose like - web browser, mobile, or back-end server.
* It's methods and functions are very easy to pick up by a total beginner. For a fact, I learnt the basics of Tensorflow in just a week following along with a <a href="">Introduction to Tensorflow for AI, ML and DL</a> course on Coursera.
* You don't necessarily need to install it to use it first. You can start building a model directly on colab, with all tools and libraries available and without any hassels.


<h3>Make Consistency your Watch-Word</h3>
<p style="text-align: center;">Consistency creates habit. Once something has become an habit, you do it naturally and it rarely every leaves.</p>

First and fore most, you should love to work in Computer Vision field. Without this love, we can't even start talking consistency. If this is the case, then I cannot stress this enough. I you want to take just ONE guideline from all the key points, let it be this one. This just tells you how important it is. You can't trade consistency for anything - even money. You want to know why I picked this? Because by applying consistency to your work in CV, you would sooner or later figure out the other helpful tips I have given you. All these tips are written to make you reading it right now understand them "early on" not that you can't figure them out for yourself. In other words, I'm trying to save you all the time you will spend by choosing the wrong paths which is a common thing for beginners. Back to consistency, what I mean is this: If you work and practice Computer Vision tutorials and projects every other day, you can achieve in months what others strived years to achieve because of their lack of consistency. I can't really do much to help you get to that level of consistency that is desirable, it's more of a personal endeavor. What I can make you undertsand is this: If you make becoming an expert in Computer Vision you Definite Chief Aim(what you want to achieve everyday more than anything), you would attain the greatest heights possible no matter how low you might start. Understand that statement, believe in it, and that's the best assistance I can render in this case.



That brings us to the end of this article on guidelines to help you take a start in the field of CV, I hope most that you find this helpful on the long run. Naturally questions would popped in your mind as you have read this comprehensive guidelines, know that your comments are very welcomed and if you please let me know if you found this helpful and what your next step would be in your quest to master Computer Vision. Thanks, Chel_
