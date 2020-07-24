---
layout: post
author: Chel
title: How to Build an Image Classification API in Tensorflow
seo: Object Detection Machinelearningmastery Keras OD Model
description: In this tutorial you would see how to apply state-of-the-art R-CNN architecture model built on the MS Coco dataset for Object Detection on a new Dataset. We use Transfer Learning for efficient and faster training epochs. You would learn how to Install the R-CNN Library, install the dataset we want to make predictions on, parse the annotation files for bounding boxes, Train Mask R-CNN Model on dataset using Transfer Learning.
img-src: ../assets\images\Blog\object_detection
---

In this tutorial you would see how to apply state-of-the-art R-CNN architecture model built on the MS Coco dataset for Object Detection on a new Dataset. We use Transfer Learning for efficient and faster training epochs. You would learn how to Install the R-CNN Library, install the dataset we want to make predictions on, parse the annotation files for bounding boxes, Train Mask R-CNN Model on dataset using Transfer Learning.


<h3>Overview</h3>
* Gentle introduction to Object Detection
* Object Detection task in Computer Vision using Mask R-CNN
* Preparing R-CNN library Model
* Installing the Dataset (Kangaroo)
* Training Mask R-CNN Model on Dataset using Transfer Learning
* Detecting Kangaroo in new Images.

Now that you have an idea of the steps taking us through this tutorial, Let's begin with it.

<h3>Introduction</h3>

Starting off in Machine Learning can be Overwhelming, I say this because I'm currently in the process of experiencing the start off altogether. There are so many fields, opinions, doubts that can go through your mind on the process of choosing a path to follow. Sometimes, actually most times, It's really cool to look at the big guys in the field that exude so much confidence showing that they know what they want. Also, there are lots of resources and articles around promising ways that would become a revelation to newcomers and map out the correct path for them. I just read some of those, and I'm more affected than when I was using my mind at a space to think for myself. My mood becomes let's say negative. But I can't be like that for long. I simply told myself I'm gonna learn them all by "doing". It's a big deal, perhaps the hardest part but with time it becomes a natural thing for you. That's the way of life. Long story short, I decided to start implementing projects that I could add to my github, LinkedIn and blog. 