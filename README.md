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
