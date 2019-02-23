# **Traffic Sign Recognition** 

## Jeyte Hagaley's Writeup

### This file is a write up of my traffic sign classifier project

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! This is the project write up with detailed information on each section provided below. 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used some basic math to compute some interesting statistics on the give data set below!

* The size of training set: 34799 samples
* The size of the validation set: 12630 samples
* The size of test set: 4410 samples
* The shape of a traffic sign image: (32, 32, 3)
* The number of unique classes/labels in the data set: 43 classes

#### 2. Include an exploratory visualization of the dataset.

The bar charts in the 3rd cell of my IPython notebook show the distribution of the different traffic signs in the data for the training, validation, and testing data sets. These images are saved in the "output images" folder and can be accessed there! 


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to YUV color space and then adjusted the image contrast. This was really helpful when trying to standardize all the images. As a last step, I normalized the image data. I combined the color space to get a final image with a shape of (32, 32, 1).

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 standardized image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 	                                   |
| Flatten	      	| Outputs 400	                                   |
| Fully connected		| Outputs 120      									|
| Dropout		|    At 50% for training, 100% for validation and testing									|
| RELU					|												|
| Fully connected		|    Outputs 43, number of classes!								|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used most of the hyperparameters from the LeNet lab. There were a few thing I changed to get much better results. The first was increasing the number of epochs using a parameter to multiple the number of epochs. I set this to be 10 to allow for higher number of epochs to be run. I ran a total of 100 epochs output the result at every 10th epoch. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy: 1.00
* validation set accuracy: 0.941
* test set accuracy of: 0.962

I used the LeNet approach we discussed in class and made a few modifications. I first removed the 2nd fully connected layer from my model. I noticed a big performance increase when I did this. The next thing I did was to add dropout after my second fully connected layer. I noticed a performance increase of about 5% to 7% when I did this. Using a much smaller Epoch size I was able to tune different parts of my model and was able to reach the validation accuracy of over 93%! When I tested my model on the test set I was constantly getting over 95% accuracy!

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify. 

All of the images I pull from the web are stored in the "test images" folder. You can also see the adjusted images in my notebook! 

The background for the images I got were not always the same as the background from images in the training set. This was my biggest worry when evaluating them against my network.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery road      		| 	Slippery road 									| 
| 	Turn right ahead   			| Speed limit (20km/h)									|
| 	No passing				| 	No passing										|
| Speed limit (70km/h)	      		| Speed limit (70km/h)					 				|
| Speed limit (60km/h)		| Speed limit (30km/h)    							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This doesn't compare too well with what my testing shows. I believe my pictures I got from the internet weren't the best quality. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook. For most of the images I provided my model seemed to be very confident in whichever image it chose, regardless of it being right or wrong. In my notebook there is all the soft maxes for each image it predicted with the top 5 soft maxes. 

