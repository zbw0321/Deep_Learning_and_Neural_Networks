# Deep_Learning_and_Neural_Networks
This are three projects about the CNN, RNN(Using LSTM) and Reinforcement Learning(Q-Learning).
---
# Project 1: CNN and Perceptron
The project is devided into several parts: 
1. the practice of perceptron Learning
2. one layer neural network
3. two layer nerual network
4. CNN

The data are using MINST dataset and the prediction accuracy are increaing by using one layer network, two layer network and CNN.

---
# Project 2: RNN using LSTM
# Introduction
The purpose of this project is developing a classifier able to detect the sentiment of movie reviews which helps a good understanding of the internal dynamics of TensorFlow and how to implement, train and test neural network.

# Methodology
In this project, there are two main parts to develop the model. 

# 1. preprocess() part
Preprocessing plays an important role in enhancing the classification accuracy. In this part, first we convert each word into lowercase, then we judge each word whether it is a punctuation, stop words and non-meaning samples, then drop it. At the beginning, we did not drop the numbers and samples, the accuracy is around 78%. Then we try to do more efforts on the pre-processing part, we drop the samples and numbers, then the test accuracy is improved to 81%. According to the above results in testing dataset, the conclusion of this step is that good pre-processing could increase the model accuracy. This is because the pre-processing could avoid to putting the whole word lists into the training model, so that there could have less irrelevant information during the model learning. 

# 2. define_graph() part
In this part, we build our training model. We choose to use Long Short-Term Memory(LSTM), this is because RNN is a kind of Neural Network used to process sequential data, but we also need to deal the gradient vanishing problem of original RNN and the LSTM could delay memory loss compared to RNN. Then, we have another problem that is over-fitting. The model has a good accuracy on the training dataset but has a low accuracy on the testing dataset. Thus, we set the dropout rate to handle the over-fitting. Finally, we do the experiments with different parameters to test the effect on model performance.

The training data and validate data are too large to upload in Github, if you have interest, please contact me.

---
# Project 3: Deep Reinforcement Learning - Q Learning using Experience Replay

# 1. Project Methodology
# (1) Neural Network Design
In this part, we design a two-layer network for training. The q_values are all available actions and we choose one action and get the Q-value computed by our neural network. Then we compute the loss in the formula: . We choose to use AdamOptimizer to be the optimizer.

# (2) Compute Target
To solve the problem of unstable learning and correlated, we use the method of Experience Replay. We build a buffer to store the previous game records and use random function to send the small
batch of training data into the neural networks for training. The size of buffer is initialized, when there are more data sending into the buffer, it will pop the previous data and insert the new data to make the buffer keep updating new data.

For computing the target, there are two conditions. If the episode has not terminated, we use the Bellman formula to get the target value, if the episode has terminated, we get the current reward for target value.

