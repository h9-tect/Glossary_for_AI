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



