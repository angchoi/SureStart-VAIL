# SureStart 2021 VAIL Program

## Responses

#### Day 1 (February 8, 2021):
During this program, I hope to gain a stronger understanding of various topics within AI and discover how they can be applied to solve challenging real-world problems. Through the daily curriculum, I aim to learn new technical skills that are needed for machine learning projects and create technology that can reflect human emotions. I hope that I can then apply my new knowledge during the Makeathon phase and learn valuable teamwork skills by working closely with my team and mentor. I am also looking forward to attending the various talks and seminars to get advice on how to have a successful career in this field!

#### Day 2 (February 9, 2021): 
1. Supervised learning is the machine learning task where the program is trained on a pre-defined set of training examples. The main goal is usually to develop a finely tuned predictor function, and the two major subcategories are regression and classification. On the other hand, with unsupervised learning, there are no training examples used. Instead, the program must find hidden patterns and relationships within unlabeled data.
2. The statement, "Scikit-Learn has the power to visualize data without a Graphviz, Pandas, or other data analysis libraries", is FALSE because Scikit-Learn is built on top of these common Python libraries. While Scikit-Learn focuses on the data modelling, these libraries are needed for the extra steps of loading, handling, manipulating, and visualising of data.

#### Day 3 (February 10, 2021):
1. Tensors are implemented in TensorFlow as multidimensional data arrays. They are the mathematical representation of physical entities with both magnitude and multiple directions, so they generalize scalars, vectors and matrices to higher dimensions. In an example of a tensor being represented by an array of 2R numbers in a 3-dimensional space, the "R" represents the rank of the tensor and the tensor will require N^R numbers in an N-dimensional space. Therefore, in a 3-dimensional space, a second-rank tensor can be represented by 3^2 (9) numbers. Tensors are used in machine learning to encode multi-dimensional data, as well as for the training and operation of deep learning models.
2. While working on the tutorial, what I noticed about computations ran in the TensorFlow programs is that the result doesn't actually get calculated. In order to actually calculate and see the result, it is necessary to run the code in an interactive Session.

#### Day 4 (February 11, 2021):
Today I learned about the difference between deep learning and machine learning, as well as how deep learning can be applied to various real-world problems. I also gained a better understanding of how artificial neural networks work and the different types of neural networks that exist. I then read the short additional article about ethics in machine learning, in which the main idea was that the three main possible avenues for "cheating" lie in the data, algorithm, and results. I believe it will be essential to constantly keep these ethical concerns in mind while working on future projects to ensure that ML systems maintain their integrity and are truly bringing about good change.

A real-world problem that could potentially be solved using deep learning is early detection of COVID-19 (and other infectious diseases). The dataset that I found for this specific problem is: https://data.mendeley.com/datasets/8h65ywd2jr/3. This dataset contains over 17,000 X-Ray and CT Chest images from both Non-COVID and COVID cases. Using this data, the deep learning algorithm that I would likely use to develop a solution is a Convolutional Neural Network. I believe that a CNN would be the best approach because they are a powerful way to learn useful representations of images and are scalable for large datasets. In addition, since CNNs are already considered the leading model for image recognition tasks, it would likely result in a higher accuracy.

#### Day 5 (February 12, 2021):
Using what I learned from the Keras Tutorial and Deep Learning Guide, I worked on developing a NN model with the News Headlines Dataset For Sarcasm Detection. The .ipynb file from my Kaggle Notebook can be found [here](https://github.com/angchoi/SureStart-VAIL/blob/main/Sarcasm-Detection.ipynb).

#### Day 8 (February 15, 2021): 
Today I learned about the different types of layers and hyperparameters that are used in convolutional neural networks, as well as the 3 main types of object recognition algorithms. I also learned how to calculate a confusion matrix and use it to gain a better understanding of the performance of a classification model.

For today's action item, I created a modified version of this [Convolutional Neural Network Kaggle Tutorial](https://www.kaggle.com/kanncaa1/convolutional-neural-network-cnn-tutorial) by extending the training data using the full training data from the [MNIST Database](http://yann.lecun.com/exdb/mnist/). To get the database of all 60,000 training examples and labels in a CSV format, I used [this Kaggle dataset](https://www.kaggle.com/oddrationale/mnist-in-csv?select=mnist_train.csv). I then evaluated the performance of the updated model (applied to the same test-set) by looking at the confusion matrix and comparing it to what was displayed in the original tutorial notebook.

Here is the link to my final Kaggle Notebook that I uploaded to this repository: https://github.com/angchoi/SureStart-VAIL/blob/main/CNN-Tutorial-for-MNIST.ipynb

#### Day 9 (February 16, 2021): 
I learned about algorithmic bias and the importance of data collection because depending on the data used to train a machine learning model, it could potentially cause the program to make decisions that further inequality and negative stereotypes. During the talk, I also learned about how implicit biases develop and influence our behaviors.
1. Machine Learning or AI concepts were utilized in the design of the "Survival of the Best Fit" game because the algorithm first reads through past applicants' CVs and gathers info about whether or not they were hired. Using large amounts of data, the model then learns what makes a candidate good or bad by doing its best to replicate the hiring decision process, which is where concerns arise about how our own human biases can appear in datasets without us even realizing it.
2. A real-world example of a biased machine learning model is the racial biases within many facial recognition systems that are used today. To make this model more fair, inclusive, and equitable, I think it is crucial to overcome the issue of a lack of diversity in the datasets, or the images that are used to train the models. By having more diverse datasets, the algorithms may be able to develop more consistent accuracies across various races of individuals. Another possible idea is using race-specific training procedures and algorithms to minimize errors. I chose this specific example of a biased model because with AI becoming increasingly crucial to the lives of the general public, it’s important that modern-day racism isn’t translated to emerging technologies, especially one like facial recognition that is utilized in such a multitude of ways (law enforcement, security, etc).

#### Day 10 (February 17, 2021): 
Today I learned about the differences between Convolutional Neural Networks & Fully Connected Neural Networks as well as the types of layers in a CNN architecture.

*Convolutional Neural Networks:*
* Specializes in image recognition and computer vision tasks
  * Since images are composed of smaller details, it analyzes each feature in isolation to make a decision about the full image 
* Types of Layers:
  * Convolutional layer: features of the image get extracted within this layer
    * a filter passes over the image and repeats the process of scanning a few pixels at a time to create a feature map
  * Pooling layer (downsampling): reduces the spatial volume of input image after convolution by maintaining only the most important information of each feature
  * Fully connected layer: takes the output of convolution/pooling and predicts the best label to describe the image
    *  goes through its own backpropagation process to determine the most accurate weights for each neuron

*Fully Connected Neural Networks:*
* Classic neural network architecture: all neurons connect to all neurons in the next layer
* Inefficient for computer vision tasks 
  * Images are very large inputs, so it would require too many connections & network parameters)

#### Day 11 (February 18, 2021):
Today I developed a CNN for classifying MNIST datasets, evaluated the performance of the final model, and used it to make predictions on new images. While working on the code, I followed [this guide](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/). Here is the link to my final Google Colab Notebook that I uploaded to this repository: https://github.com/angchoi/SureStart-VAIL/blob/main/CNN-for-MNIST-Classification.ipynb

#### Day 12 (February 19, 2021):
Today I used [this guide](https://keras.io/examples/vision/image_classification_from_scratch/) to work on the trask of training an image classifier from scratch on the Kaggle Cats vs Dogs binary classification dataset. My final Google Colab Notebook can be found here: https://github.com/angchoi/SureStart-VAIL/blob/main/Image_Classification_from_Scratch.ipynb.

#### Day 15 (February 22, 2021):
Today I gained a greater understanding of machine learning ethics & the different types of bias, as well as the importance of context in data collection and algorithm implementation. I also learned about various tools that can be used to assess bias and help promote transparency. 

I then worked on a Gender Classification Model using [this dataset](https://www.kaggle.com/thanaphatj/gender-classification-of-facial-images-cnn/#data) of facial images and [this Kaggle notebook](https://www.kaggle.com/thanaphatj/gender-classification-of-facial-images-cnn) as a guide. My final code can be found here: https://github.com/angchoi/SureStart-VAIL/blob/main/Gender-Classification-of-Facial-Images.ipynb

#### Day 16 (February 23, 2021):
The rectified linear activation function (ReLU) is a piecewise linear function that will output the input directly if it is positive and output zero otherwise. It has become the default activation function when developing many types of neural networks, as it overcomes the vanishing gradient problem and therefore allows models to learn faster and achieve better performance. One use case of ReLU is with Convolutional Neural Networks (CNNs) in order to help increase the non-linearity in images. It can be used as the activation function on the filter maps and pooling layer that follows. In general, the main advantages of using ReLU include:
1. Cheaper Computations: requires just a simple max() function; no need for the use of an exponential calculation like with tanh and sigmoid
2. Spare Representation: the activation of hidden layers can contain one or more true zero values (this is a desirable property as it can speed up learning and simplify the model)
3. Linear Behavior: models are easier to optimize
4. Train Deep Networks: "reach their best performance without requiring any unsupervised pre-training on purely supervised tasks with large labeled datasets"

#### Day 17 (February 24, 2021):
Today I  learned about how to choose and implement various loss functions for regression and binary classification, especially the mean squared error loss. I also learned how to incorporate L2 regularization in our models while working on creating a simple CNN to predict house prices. The dataset used can be found [here](https://drive.google.com/file/d/1GfvKA0qznNVknghV4botnNxyH-KvODOC/view), and the tutorial that I followed is [here](https://hackernoon.com/build-your-first-neural-network-to-predict-house-prices-with-keras-3fb0839680f4). 

The link to my final Google Colab Notebook is: https://github.com/angchoi/SureStart-VAIL/blob/main/Predict_House_Prices.ipynb.

#### Day 18 (February 25, 2021):
Today I learned about handling overfitting in deep learning models using the [Twitter US Airline Sentiment data set](https://www.kaggle.com/crowdflower/twitter-airline-sentiment). I used [this hands-on sentiment analysis tutorial](https://towardsdatascience.com/handling-overfitting-in-deep-learning-models-c760ee047c6e#:~:text=Overfitting%20occurs%20when%20you%20achieve,are%20irrelevant%20in%20other%20data.&text=The%20best%20option%20is%20to%20get%20more%20training%20data), which included three different approaches for achieving better generalization. 

My code can be found here: https://github.com/angchoi/SureStart-VAIL/blob/main/Handling_Overfitting_with_Twitter_Data.ipynb.

I also went back to my housing prices model from yesterday and changed the loss to regression based functions. I observed that using mean squared error resulted in the loss curves for training and validation data to be slightly closer together than the original loss function of binary cross entropy. The loss curves for training & validation then got even closer (almost completely overlapping) when I used mean absolute error instead.

#### Day 19 (February 26, 2021):
I first worked on [this Upsampling Tutorial](https://machinelearningmastery.com/upsampling-and-transpose-convolution-layers-for-generative-adversarial-networks/) about adding upscaling layers to a deep learning model. In particular, I discovered how to use the UpSampling 2D Layer (simply doubles the dimensions of the input) and the Conv2DTranspose Layer (performs an inverse convolution operation) when generating images. I have uploaded my code to this repository here: https://github.com/angchoi/SureStart-VAIL/blob/main/Upsampling_Tutorial.ipynb

Then, I followed [this Autoencoder Tutorial](https://blog.keras.io/building-autoencoders-in-keras.html) with actual code and visualization for autoencoder based reconstruction and noise removal. I was able to work on code examples of the following models: a simple autoencoder based on a fully-connected layer, sparse autoencoder, deep fully-connected autoencoder, deep convolutional autoencoder, image denoising model, sequence-to-sequence autoencoder, and variational autoencoder. The link to my code is here: 

#### Day 22 (March 1, 2021):
Today looked through [this beginner friendly guide](https://towardsdatascience.com/a-list-of-beginner-friendly-nlp-projects-using-pre-trained-models-dc4768b4bec0) with six different Natural Language Processing projects using pre-trained models. I chose to work on Project #4, Language identifier. *need to add code*

I then learned about the ethical implications of big NLP models such as GPT-2 through [this article](https://openai.com/blog/better-language-models/).

#### Day 23 (March 2, 2021):
I worked on an emotion detection project using OpenCV and Keras by following [this tutorial](https://medium.com/swlh/emotion-detection-using-opencv-and-keras-771260bbd7f7). *need to add code*
