Multi-Class Classifier With Softmax (Manual Forward Pass)

Project Overview

This project implements a manual multi-class classification neural network using pure Python.
No external libraries such as NumPy, TensorFlow, or PyTorch are used.

The model performs:

Forward pass through a small neural network

ReLU activation in the hidden layer

Softmax activation in the output layer

Categorical Cross-Entropy (CCE) loss calculation

Prediction using argmax

Average loss calculation

Printing of all intermediate values (z1, a1, z2, a2)

This project helps build a clear understanding of how neural networks work internally.

Aim

To manually build a small neural network that performs multi-class classification using:

ReLU activation in the hidden layer

Softmax activation in the output layer

Categorical Cross-Entropy loss function

Objectives

Understand how Softmax produces class probabilities.

Implement ReLU activation for non-linearity.

Compute forward propagation manually.

Calculate Categorical Cross-Entropy (CCE) loss for each sample.

Display intermediate values: z1, a1, z2, a2.

Predict classes using argmax.

Compute average loss across samples.

Build foundation for backpropagation and training in future projects.

Network Architecture
Input Layer (2 features)
        ↓
Hidden Layer (2 neurons, ReLU)
        ↓
Output Layer (3 neurons, Softmax)

Forward Pass Explanation

For each input sample:

Hidden layer linear output:
z1 = W1 * x + b1

Hidden layer activation:
a1 = ReLU(z1)

Output layer linear output:
z2 = W2 * a1 + b2

Output activation:
a2 = Softmax(z2)

Prediction:
predicted_class = argmax(a2)

Loss (Categorical Cross-Entropy):
Loss = -Σ(y_true * log(y_pred))

Dataset Used

Three samples with 2 features each and one-hot encoded labels:

X = [[1.0, 2.0],
     [0.5, -1.0],
     [2.0, 1.0]]

Y = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

Technologies Used

Python (pure)

Built-in math library only

How to Run This Project
Step 1: Save the Script

Create a file named:

multi_class_classifier.py


Paste the Python code inside it.

Step 2: Run the Program

Open a terminal or command prompt and run:

python multi_class_classifier.py

Step 3: View Output

The program will display:

Input values

Hidden layer outputs (z1 and a1)

Final layer outputs (z2 and a2)

Predicted class

Actual class

Loss for each sample

Final average loss

Real-Life Applications

Multi-class classification is used in many real systems, including:

Face Recognition

Email Spam Detection

Photo Object Identification

Voice Assistants

Medical Diagnosis Models

Traffic Prediction

Product Recommendation Systems

Conclusion

This project demonstrates:

How neural networks compute outputs manually

How Softmax converts scores into probabilities

How Categorical Cross-Entropy measures prediction accuracy

How a neural network predicts multi-class outputs step by step

It provides a strong foundation for understanding deeper neural network concepts such as training, backpropagation, and gradient descent.
