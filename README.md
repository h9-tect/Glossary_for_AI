# Math Glossary

**Hessian**  
A square matrix of second-order partial derivatives of a scalar-valued function, which is used in optimization to determine the curvature of multivariable functions.

**Random Variable**  
A variable whose possible values are numerical outcomes of a random phenomenon, categorized into discrete (finite or countable outcomes) and continuous (infinite outcomes).

**Matrix**  
A rectangular array of numbers, symbols, or expressions, arranged in rows and columns, used in linear algebra to represent systems of linear equations and perform various matrix operations.

**Entropy**  
In information theory, entropy is a measure of the unpredictability or randomness of a data source, indicating the amount of uncertainty in the next piece of information.

**Mutual Information**  
A measure that quantifies the amount of information obtained about one random variable through another, reflecting the reduction in uncertainty for one variable given a known value of another.

**Dot Product**  
An algebraic operation that takes two equal-length sequences of numbers (usually coordinate vectors) and returns a single number, used to determine angles between vectors or calculate physical quantities like work.

**Mean**  
The average value of a dataset, calculated as the sum of all data points divided by the number of points, representing the central tendency of the data.

**Variance**  
A statistical measure of the dispersion of data points in a dataset, calculated as the average of the squared differences from the Mean.

**L2 Norm**  
Also known as the Euclidean norm, it represents the distance from the origin to the point in a Euclidean space, calculated as the square root of the sum of the squares of its components.

**Chain Rule (Differentiation)**  
A fundamental calculus rule for finding the derivative of composite functions, stating that the derivative is the product of the derivatives of the composed functions.

**Fourier Transform**  
A mathematical transform that decomposes functions depending on space or time into functions depending on frequency, used extensively in signal analysis and physics.

**Continuity**  
A function is continuous if, at every point in the domain, the limit of the function as it approaches the point equals the function's value at that point.

**Lipschitz Continuity**  
A function satisfies Lipschitz continuity if there exists a constant such that, for all point pairs in its domain, the absolute difference in function values is bounded by the product of the constant and the distance between points.

**Chain Rule (Probability)**  
A probability theory rule that allows the computation of the joint distribution of a set of random variables using only conditional probabilities.

**Polynomial**  
An expression consisting of variables and coefficients, involving only the operations of addition, subtraction, multiplication, and non-negative integer exponentiation.

**Cantor's Diagonal Argument**  
A mathematical proof demonstrating that there are infinite sets which cannot be put into one-to-one correspondence with the infinite set of natural numbers.

**Jacobian**  
A matrix of all first-order partial derivatives of a vector-valued function, important for transformations of coordinates and differential equations.

**Linear Operator**  
A mapping between two vector spaces that preserves the operations of vector addition and scalar multiplication.

**Gradient**  
In vector calculus, the gradient of a scalar function is a vector field that points in the direction of the greatest rate of increase of the function.

**Bayes' Theorem**  
A fundamental theorem in probability theory that describes the probability of an event based on prior knowledge of conditions that might be related to the event.

**Vector**  
An object with both magnitude and direction, used to describe quantities in physics and engineering.

**Joint Law, Product Law**  
The joint law refers to the joint probability distribution of two or more random variables, and the product law is a rule to find the probability that two events both occur.

**Gaussian Distribution**  
Also known as the normal distribution, it's a probability distribution that describes how the values of a variable are distributed. It is the basis for the bell-shaped curve.

**Distribution**  
In statistics, a distribution is a function that shows the possible values for a variable and the frequency of these values.

**Determinant**  
A scalar value that can be computed from the elements of a square matrix and encodes properties like the matrix's invertibility.

**Rank**  
The rank of a matrix is defined as the maximum number of linearly independent row or column vectors in the matrix.

**Eigen-decomposition**  
The process of decomposing a matrix into eigenvalues and eigenvectors, which is crucial for solving systems of linear equations, among other applications.

**SVD**  
Singular value decomposition, a method of decomposing a matrix into three matrices, exposing many of the useful properties of the original matrix.

**Maximum Likelihood**  
A method of estimating the parameters of a statistical model, which maximizes the likelihood that the process described by the model produces the observed data.

**Central Limit Theorem**  
A statistical theory that states when independent random variables are added, their normalized sum tends toward a normal distribution, regardless of the original variables' distribution.

# Computer Science Glossary


### Polymorphism
Polymorphism in computer science refers to the ability of a function or an object to take on many forms. It allows methods to do different things based on the object it is acting upon, essential for achieving abstraction and flexibility in object-oriented programming.

### Recursion
Recursion is a programming technique where a function calls itself directly or indirectly. It's often used to solve problems that can be broken down into simpler, repetitive tasks. It is an effective method for tasks such as traversing data structures like trees and graphs.

### Value Passed by Reference
Passing by reference means that a function receives a reference to the variable itself, rather than a copy of its value. This allows the function to modify the original variable's value directly.

### Binary Search
Binary search is an efficient algorithm for finding an item from a sorted list of items. It works by repeatedly dividing in half the portion of the list that could contain the item, until you've narrowed the possible locations to just one.

### Quick Sort
Quick sort is a divide-and-conquer algorithm that selects a 'pivot' element from the array and partitions the other elements into two sub-arrays, according to whether they are less than or greater than the pivot. The sub-arrays are then sorted recursively.

### Parallel Scan
Parallel scan, also known as parallel prefix sum, is an algorithm used to compute prefix sums efficiently in parallel. It's useful in applications where you need to perform computations on large data arrays in a parallel computing environment.

### Mutability
In computer programming, mutability refers to the ability of an object to be changed after it has been instantiated. Mutable objects have their fields and states changeable, while immutable objects do not allow any alteration after creation.

### Turing Machine
A Turing machine is a mathematical model of computation that defines an abstract machine which manipulates symbols on a strip of tape according to a table of rules. Despite the simplicity of the model, Turing machines can simulate any computer algorithm.

### FP32
FP32 refers to a 32-bit floating-point data type used in computing. It is a standard format for representing and manipulating real numbers in digital computers, particularly in graphics processing and deep learning applications.

### Iterator
An iterator is an object that enables a programmer to traverse a container, particularly lists. Various types of iterators are capable of accessing the data elements of a container in a sequential manner.

### Interpreter vs Compiler
An interpreter directly executes instructions written in a programming or scripting language without previously converting them to an object code or machine code. A compiler, on the other hand, transforms high-level code into machine code that the computer's processor can execute directly.

### Anonymous Function
Also known as lambda functions, anonymous functions are functions that are defined without a name. They are often used for constructing short, ad-hoc functions, and are typically arguments being passed to higher-order functions.

### Set
In computer science, a set is an abstract data type that can store unique values, without any particular order. It is typically used to ensure that no duplicates are entered.

### Binary Heap
A binary heap is a complete binary tree which satisfies the heap ordering property. It can be seen as a binary tree with two additional constraints: the shape property and the heap property, often used in priority queues.

### Mutex
A mutex (mutual exclusion object) is a program object that allows multiple program threads to share the same resource, such as file access, but not simultaneously. When a program is locked by a mutex, no other thread can access the locked region of code until the mutex is unlocked.

### Cache Memory
Cache memory is a small-sized type of volatile computer memory that provides high-speed data storage and access to the processor and stores frequently used data and instructions.

### Scope of a Variable or Function
Scope refers to the visibility of variables and functions in parts of a program. It determines the accessibility of these variables and functions to various parts of the code.

### Dynamic Programming
Dynamic programming is a method for solving complex problems by breaking them down into simpler subproblems. It is applicable where the subproblems are not independent, i.e., when subproblems share subsubproblems. In this context, the technique of storing solutions to subproblems (instead of recomputing them) is used.

### Hash Table
A hash table, or a hash map, is a data structure that implements an associative array abstract data type, a structure that can map keys to values. A hash table uses a hash function to compute an index into an array of buckets or slots, from which the desired value can be found.

### Big-O Notation
Big-O notation is a mathematical notation that describes the limiting behavior of a function when the argument tends towards a particular value or infinity. It is a primary tool in computational complexity theory to classify algorithms according to how their run time or space requirements grow as the input size grows.

### Turing Complete
A system of data-manipulation rules (such as a computer's instruction set, a programming language, or a cellular automaton) is Turing complete if it can be used to simulate any Turing machine. This concept is used to determine the computational equivalence of different systems.

### Class Inheritance
Class inheritance is a feature in object-oriented programming that allows a class to inherit properties and behaviors (methods) from another class. Referred to as a "subclass" or "derived class," the class that inherits is able to extend the functionality of the "base class" or "superclass."

### Closure
In programming, a closure is a technique for implementing lexically scoped name binding in a language with first-class functions. Operationally, a closure is a record storing a function together with an environment that binds each of the function's non-local variables to its corresponding value.

### Loop Unrolling
Loop unrolling is an optimization technique that involves duplicating the body of a loop a certain number of times, reducing the overhead of loop control and increasing the speed of the program.

### Complexity
In computational theory, complexity refers to the amount of resources required for the execution of an algorithm. The more steps (time complexity) or memory (space complexity) an algorithm requires to complete, the more complex it is considered.

# Machine Learning Glossary


### VC Dimension
The Vapnik-Chervonenkis (VC) dimension is a measure of the capacity of a statistical classification algorithm, defined as the cardinality of the largest set of points that the algorithm can shatter. It is used to quantify the complexity and capacity of a model to learn various functions.

### Over-fitting, Under-fitting
**Over-fitting** occurs when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. **Under-fitting** occurs when a model is too simple, both in terms of not capturing the underlying pattern of the data and not performing well on new data.

### Logistic Regression
Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model.

### Q-value
In reinforcement learning, a Q-value is a measure of the value of taking a particular action in a particular state, according to what the model currently knows about the environment. Q-values are used in certain types of learning algorithms to iteratively improve the behavior of the learning agent.

### Kernel Trick
The kernel trick is a method used in machine learning algorithms that enables them to operate in a high-dimensional, implicit feature space without ever computing the coordinates of the data in that space, but instead by computing the inner products between the images of all pairs of data in the feature space.

### Boosting
Boosting is an ensemble technique that attempts to create a strong classifier from a number of weak classifiers. It is done by building a model from the training data, then creating a second model that attempts to correct the errors from the first model.

### PCA (Principal Component Analysis)
Principal Component Analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.

### Feature Design
Feature design, also known as feature engineering, is the process of using domain knowledge of the data to create features that make machine learning algorithms work. It is fundamental to the application of machine learning, and involves an aspect of art as much as science.

### Linear Regression
Linear regression is a linear approach to modelling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables).

### Expectation-Maximization, GMM
The **Expectation-Maximization (EM)** algorithm is an iterative method to find maximum likelihood or maximum a posteriori (MAP) estimates of parameters in statistical models, where the model depends on unobserved latent variables. **Gaussian Mixture Models (GMMs)** are a type of model used in statistical data modeling involving multiple sub-populations, assuming that each sub-population follows a Gaussian distribution.

### SVM (Support Vector Machine)
Support Vector Machine (SVM) is a supervised machine learning algorithm which can be used for both classification or regression challenges. It performs classification by finding the hyperplane that best divides a dataset into classes.

### Bellman Equation
The Bellman equation, also known as dynamic programming equation, describes the principle of optimality for dynamic systems. It plays a crucial role in reinforcement learning and optimal control.

### Decision Tree
A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility.

### Train/Validation/Test Sets
These sets are used to evaluate how well a machine learning model will perform on independent data. **Training set** is used to train the model, **validation set** is used to tune the parameters of the model, and **test set** is used to test the performance of the model.

### Naive Bayesian Model
Naive Bayes classifiers are a family of simple probabilistic classifiers based on applying Bayes' theorem with strong (naive) independence assumptions between the features.

### Autoregressive Model
An autoregressive model is a representation of a type of random process; as such, it is used to describe certain time-varying processes in nature, economics, etc.

### Bias-Variance Dilemma
The bias-variance dilemma is the problem of simultaneously minimizing two sources of error that prevent supervised learning algorithms from generalizing beyond their training set: the bias, which is the error caused by approximations, and the variance, which is the error caused by random fluctuations in the training set.

### Policy Gradient
Policy gradient methods are a class of reinforcement learning techniques that rely upon optimizing parametrized policies with respect to the expected return by gradient descent.

### Random Forest
Random Forest is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

### k-NN (k-Nearest Neighbors)
The k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space.

### Perceptron Algorithm
The perceptron is a simple algorithm for binary classifiers that is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector.


# Deep Learning Glossary


### Adam
Adam (Adaptive Moment Estimation) is an optimization algorithm that combines the advantages of two other extensions of stochastic gradient descent: Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp). It is designed to handle sparse gradients on noisy problems.

### Softmax
The softmax function, or normalized exponential function, is a generalization of the logistic function that "squashes" a K-dimensional vector of arbitrary real values to a K-dimensional vector of real values in the range (0, 1) that add up to 1. It is used in multinomial logistic regression and is often the last operation done in a network.

### Residual Connections
Residual connections, or shortcuts, help to avoid the vanishing gradient problem in deep networks by allowing gradients to flow through a shortcut path. They are used extensively in deep residual networks (ResNets).

### Autograd
Autograd is a Python library that automatically differentiates native Python and Numpy code. It provides a flexible and efficient system for automatically computing gradients.

### ReLU
Rectified Linear Unit (ReLU) is a type of activation function that is defined as the positive part of its argument, where \(f(x) = max(0, x)\). ReLU is popular because it speeds up training without a significant penalty to generalization accuracy.

### Dropout
Dropout is a technique for addressing overfitting. The key idea is to randomly drop units (along with their connections) from the neural network during training. This prevents units from co-adapting too much.

### CLIP
CLIP (Contrastive Language–Image Pre-training) by OpenAI leverages large amounts of data and computational resources to link images with texts in a way that enables zero-shot classification of images by using natural language.

### Xavier's Initialization
Also known as Glorot initialization, this technique provides an effective way to initialize the weights of a neural network to help prevent the vanishing or exploding gradient problems during training.

### Vanishing Gradient
The vanishing gradient problem occurs when the gradient of the loss function is propagated backward through the network, and the gradient can become vanishingly small, effectively preventing the weights from changing its value.

### LeNet
Developed by Yann LeCun, LeNet was one of the first convolutional neural networks which efficiently applied the gradient-based learning methods to digital images. It was primarily developed for handwriting and character recognition.

### ViT
Vision Transformer (ViT) applies the principles of the transformer architecture, typically used for natural language processing, to image classification tasks.

### Transposed Convolution Layer
Transposed convolutions are used in convolutional neural networks to upsample input feature maps, often as part of a process to increase the resolution of images in tasks like image segmentation.

### Checkpoint (during the forward pass)
Creating checkpoints during the forward pass of training a neural network helps in saving the model's state at intermittent points. This can be used for recovery if the training process is interrupted, or for starting subsequent training runs from a particular state.

### Minibatch
In neural network training, a minibatch is a subset of the training dataset used to train the model for a single iteration. This allows for both efficient computation and more stable convergence in stochastic gradient descent.

### Masked Model
In models such as transformers, masking involves hiding certain elements of the input from the model; this is a critical part of training language models to handle arbitrary input sequences efficiently.

### Supervised / Unsupervised
Supervised learning algorithms are trained using labeled data, while unsupervised learning algorithms are trained using unlabeled data, which allows them to find structure in the input data on their own.

### Data Augmentation
Data augmentation involves artificially increasing the diversity of data available for training models without actually collecting new data. Techniques include rotation, cropping, resizing, and color modification among others.

### Attention Block
In deep learning, especially in models like transformers, attention blocks help the model focus on relevant parts of the input data, improving the accuracy and efficiency of tasks like language translation.

### SGD
Stochastic Gradient Descent (SGD) is a simple yet highly effective approach to fitting linear classifiers and regressors under convex loss functions. It's particularly useful for large-scale and sparse machine learning problems.

### Batchnorm
Batch normalization is a technique to provide any layer in a neural network with inputs that are zero mean/unit variance, and by doing so, helps in speeding up the training of neural networks.

### Gradient Clipping
Gradient clipping involves limiting (or "clipping") the values of gradients during backpropagation to address the exploding gradients problem in training deep neural networks.

### Tokenizer
A tokenizer processes text to break it down into its constituent parts (tokens), which are often words or subwords, making the text more amenable to deep learning models.

### VAE
Variational Autoencoders (VAE) are a type of generative model that use a probabilistic approach to describe observations in a latent space. Unlike standard autoencoders, VAEs are designed to impose a distribution on the latent variables.

### Weight Decay
Weight decay is a regularization technique used to add a small penalty, which is a fraction of weight magnitude, to the loss function. The penalty term is proportional to the square of the value of the weights.

### GELU
The Gaussian Error Linear Unit (GELU) is a non-linear activation function used in deep learning that models the data better on complex tasks like natural language understanding.

### LSTM, GRU
Long Short-Term Memory (LSTM) units and Gated Recurrent Units (GRU) are types of RNN architectures designed to help capture information over extended sequences, useful in applications like time series prediction and natural language processing.

### GAN
Generative Adversarial Networks (GANs) consist of two neural networks, a generator and a discriminator, that are trained simultaneously. The generator's goal is to produce data indistinguishable from real data, while the discriminator tries to differentiate between real and generated data.

### ResNet
Residual Networks (ResNet) are a type of convolutional neural network (CNN) where connections skip one or more layers. They are designed to enable training of extremely deep neural networks to achieve better performance.

### Straight-Through Estimator
The Straight-Through Estimator is a method for backpropagation through discrete variables. It is commonly used for training models that include non-differentiable operations, like quantization.

### Convolution Layer
Convolutional layers are the building blocks of convolutional neural networks (CNNs). These layers apply a convolution operation to the input, passing the result to the next layer.

### Pre-training / Fine-tuning
Pre-training refers to training a neural network model on a large dataset with a general-purpose task before it is fine-tuned on a specific task. This helps in improving the performance of the neural network on tasks that have smaller datasets.

### Perplexity
Perplexity is a measure of how well a probability model predicts a sample, often used to evaluate language models. A lower perplexity score indicates better predictive performance.

### Logits
In the context of neural networks, logits are the raw outputs of the last layer in the network before applying the activation function like softmax.

### CLS Token
In models such as BERT, the CLS token is added to the beginning of each input example and is used as the aggregate representation for classification tasks.

### Forward Pass
The forward pass in a neural network refers to the process of computing the output of the network for a given input by passing it through each layer in turn.

### Transformer (original one), GPT
The Transformer model is a type of architecture that relies solely on attention mechanisms to draw global dependencies between input and output. The Generative Pre-trained Transformer (GPT) is a version of Transformer with a specific training method that uses unsupervised learning, which then can be fine-tuned with supervised learning tasks.

### Backward Pass
The backward pass in neural networks is the process where gradients are computed for network parameters, typically used for training via backpropagation.

### Autoencoder, Denoising Autoencoder
An autoencoder is a type of neural network used to learn efficient encodings. The denoising autoencoder is a variant that is trained to use the encoded representation to reconstruct the input from a corrupted version of it.

### Layer Norm
Layer normalization is a technique that normalizes the inputs across the features instead of the batch in neural networks. It is particularly useful in recurrent neural networks.

### GNN
Graph Neural Networks (GNNs) are designed to process data that is represented as graphs, enabling new forms of learning where the model can take graph structure into account.

### Learning Rate Schedule
A learning rate schedule modifies the learning rate during training, typically reducing the rate according to a predefined schedule. This helps in optimizing the training process.

### Diffusion Model
Diffusion models are a type of generative model that constructs data by starting from noise and gradually shaping it towards the data distribution through a process known as denoising.

### Cross-Entropy
Cross-entropy is a measure from the field of information theory, building upon entropy and generally used to quantify the difference between two probability distributions.

### Max Pooling, Average Pooling
Pooling layers downsample each feature map in the network, reducing the dimensions and allowing for assumptions to be made about features contained in the sub-regions binned. Max pooling and average pooling are techniques to perform this downsampling by taking the maximum and average value, respectively, in each patch of the feature map.

### RNN
Recurrent Neural Networks (RNNs) are a class of neural networks where connections between units form a directed cycle. This creates an internal state of the network which allows it to exhibit dynamic temporal behavior.

### Contrastive Loss
Contrastive loss is used in learning algorithms that aim to learn from the similarity and dissimilarity between pairs of samples, commonly used in tasks like image captioning and face verification.

### Positional Encoding
Positional encoding is used in Transformer models to give the model information about the relative or absolute positioning of the tokens in the input sequences.

### Causal Model
Causal models in deep learning are used to understand and manage the cause and effect relationships. They are crucial for decision-making processes where predictions or simulations of potential outcomes are needed.

### Attention Layer
An attention layer is a component in neural network architectures that weighs the input features of the data based on their importance to the task at hand. It is a key component of models that require focus on specific aspects of the input data, such as models that involve natural language processing.

### SSL (Self-Supervised Learning)
Self-supervised learning is a form of unsupervised learning where the data provides the supervision. It involves creating a pretext task, such as predicting the next word in a sentence, to help the model learn the proper context and features from the data.

### MSE (Mean Squared Error)
Mean Squared Error (MSE) is a common loss function used in regression problems. It measures the average squared difference between the estimated values and what is estimated.

### Tensor
In the context of machine learning, a tensor is a generalization of vectors and matrices to potentially higher dimensions, and is a primary data structure used in neural networks.

