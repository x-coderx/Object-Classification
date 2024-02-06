# Deep Neural Network for Image Classification

### Problem Statement: 
We have a dataset ("data.h5") containing:

- a training set of m_train images labelled as cat (1) or non-cat (0)
- a test set of m_test images labelled as cat and non-cat
- each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB).

### Pre-Requisites :
#### For Mac user:
Create a project directory and open a terminal instance in project directory. Run following commands to setup environment and pre-requisites.
- Get source code : git clone https://github.com/x-coderx/ObjectClassification.git
- Install Python : brew install python@3.9
- Create your virtual environment : python3.9 -m venv venv
- Activating your virtual environment : source venv/bin/activate
- Install Requirements : python3.9 -m pip install -r requirements.txt
- Finally, run "Deep Neural Network - Application v8.ipynb"

#### For Windows user:
Create a project directory and open a cmd instance in project directory. Run following commands to setup environment and pre-requisites.
- Get source code : git clone https://github.com/x-coderx/ObjectClassification.git
- Download latest python version from : https://www.python.org/ftp/python/3.12.1/python-3.12.1-macos11.pkg and install.
- Create your virtual environment : python -m venv venv
- Activating your virtual environment : venv\scripts\activate
- Install Requirements : python -m pip install -r requirements.txt
- Finally, run "Deep Neural Network - Application v8.ipynb"


### Understanding Image :
We have 64 X 64 total pixels for each image and every pixel is a combination of RGB colors. Which implies that our input image can be depicted as vector of shape (64,64,3). Since each pixel is an important feature (similalry each color contibution is an important feature) which will determine Cat or not Cat.
Idea is to flatten our array to shape of (n_x,1) for each image and arrange them vertically for all training set of shape (n_x,m).

![](images/imvectorkiank.png)

### Architecture of our model :
We will build two different models:
- A 2-layer neural network
- An L-layer deep neural network

We will then compare the performance of these models, and also try out different values for $L$. 


### General methodology

We will follow the Deep Learning methodology to build the model:
1. Initialize parameters / Define hyperparameters
2. Loop for num_iterations:
    - Forward propagation
    - Compute cost function
    - Backward propagation
    - Update parameters (using parameters, and grads from backprop) 
3. Use trained parameters to predict labels

## Two-layer neural network

Using the helper functions implementing a 2-layer neural network with the following structure: *LINEAR -> RELU -> LINEAR -> SIGMOID*.
```python
def initialize_parameters(n_x, n_h, n_y):
    ...
    return parameters 
def linear_activation_forward(A_prev, W, b, activation):
    ...
    return A, cache
def compute_cost(AL, Y):
    ...
    return cost
def linear_activation_backward(dA, cache, activation):
    ...
    return dA_prev, dW, db
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
```
# Result: 
Our 2-layer neural network has (72%) accuracy.

## L-layer Neural Network

Using the helper functions to build our L-layer neural network with the following structure: *[LINEAR -> RELU] X (L-1) -> LINEAR -> SIGMOID*.
```python
def initialize_parameters_deep(layers_dims):
    ...
    return parameters 
def L_model_forward(X, parameters):
    ...
    return AL, caches
def compute_cost(AL, Y):
    ...
    return cost
def L_model_backward(AL, Y, caches):
    ...
    return grads
def update_parameters(parameters, grads, learning_rate):
    ...
    return parameters
```

# Result : 
Our 4-layer neural network has better performance (80%) than our 2-layer neural network (72%) on the same test set.
