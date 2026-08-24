
<H3>NAME: PREM KUMAR S</H3>
<H3>EREGISTER NO: 212223240125</H3>
<H3>EX. NO.4</H3>
<H3>DATE:20.08.2026</H3>
<H1 ALIGN =CENTER>Implementation of MLP with Backpropagation for Multiclassification</H1>
<H3>Aim:</H3>
To implement a Multilayer Perceptron for Multi classification
<H3>Theory</H3>

A multilayer perceptron (MLP) is a feedforward artificial neural network that generates a set of outputs from a set of inputs. An MLP is characterized by several layers of input nodes connected as a directed graph between the input and output layers. MLP uses back propagation for training the network. MLP is a deep learning method.
A multilayer perceptron is a neural network connecting multiple layers in a directed graph, which means that the signal path through the nodes only goes one way. Each node, apart from the input nodes, has a nonlinear activation function. An MLP uses backpropagation as a supervised learning technique.
MLP is widely used for solving problems that require supervised learning as well as research into computational neuroscience and parallel distributed processing. Applications include speech recognition, image recognition and machine translation.
 
MLP has the following features:

Ø  Adjusts the synaptic weights based on Error Correction Rule

Ø  Adopts LMS

Ø  possess Backpropagation algorithm for recurrent propagation of error

Ø  Consists of two passes

  	(i)Feed Forward pass
	         (ii)Backward pass
           
Ø  Learning process –backpropagation

Ø  Computationally efficient method

![image 10](https://user-images.githubusercontent.com/112920679/198804559-5b28cbc4-d8f4-4074-804b-2ebc82d9eb4a.jpg)

3 Distinctive Characteristics of MLP:

Ø  Each neuron in network includes a non-linear activation function

![image](https://user-images.githubusercontent.com/112920679/198814300-0e5fccdf-d3ea-4fa0-b053-98ca3a7b0800.png)

Ø  Contains one or more hidden layers with hidden neurons

Ø  Network exhibits high degree of connectivity determined by the synapses of the network

3 Signals involved in MLP are:

 Functional Signal

*input signal

*propagates forward neuron by neuron thro network and emerges at an output signal

*F(x,w) at each neuron as it passes

Error Signal

   *Originates at an output neuron
   
   *Propagates backward through the network neuron
   
   *Involves error dependent function in one way or the other
   
Each hidden neuron or output neuron of MLP is designed to perform two computations:

The computation of the function signal appearing at the output of a neuron which is expressed as a continuous non-linear function of the input signal and synaptic weights associated with that neuron

The computation of an estimate of the gradient vector is needed for the backward pass through the network

TWO PASSES OF COMPUTATION:

In the forward pass:

•       Synaptic weights remain unaltered

•       Function signal are computed neuron by neuron

•       Function signal of jth neuron is
            ![image](https://user-images.githubusercontent.com/112920679/198814313-2426b3a2-5b8f-489e-af0a-674cc85bd89d.png)
            ![image](https://user-images.githubusercontent.com/112920679/198814328-1a69a3cd-7e02-4829-b773-8338ac8dcd35.png)
            ![image](https://user-images.githubusercontent.com/112920679/198814339-9c9e5c30-ac2d-4f50-910c-9732f83cabe4.png)



If jth neuron is output neuron, the m=mL  and output of j th neuron is
               ![image](https://user-images.githubusercontent.com/112920679/198814349-a6aee083-d476-41c4-b662-8968b5fc9880.png)

Forward phase begins with in the first hidden layer and end by computing ej(n) in the output layer
![image](https://user-images.githubusercontent.com/112920679/198814353-276eadb5-116e-4941-b04e-e96befae02ed.png)


In the backward pass,

•       It starts from the output layer by passing error signal towards leftward layer neurons to compute local gradient recursively in each neuron

•        it changes the synaptic weight by delta rule

![image](https://user-images.githubusercontent.com/112920679/198814362-05a251fd-fceb-43cd-867b-75e6339d870a.png)

<H3>Algorithm:</H3>

1. Import the necessary libraries of python.

2. After that, create a list of attribute names in the dataset and use it in a call to the read_csv() function of the pandas library along with the name of the CSV file containing the dataset.

3. Divide the dataset into two parts. While the first part contains the first four columns that we assign in the variable x. Likewise, the second part contains only the last column that is the class label. Further, assign it to the variable y.

4. Call the train_test_split() function that further divides the dataset into training data and testing data with a testing data size of 20%.
Normalize our dataset. 

5. In order to do that we call the StandardScaler() function. Basically, the StandardScaler() function subtracts the mean from a feature and scales it to the unit variance.

6. Invoke the MLPClassifier() function with appropriate parameters indicating the hidden layer sizes, activation function, and the maximum number of iterations.

7. In order to get the predicted values we call the predict() function on the testing data set.

8. Finally, call the functions confusion_matrix(), and the classification_report() in order to evaluate the performance of our classifier.

<H3>Program:</H3>

```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Load Dataset (Direct download for Google Colab)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]
diabetes_df = pd.read_csv(url, names=columns)

# 2. Explore & Clean Zero Values
print("Initial Shape:", diabetes_df.shape)
diabetes_df = diabetes_df[diabetes_df['BloodPressure'] != 0]
diabetes_df = diabetes_df[diabetes_df['Insulin'] != 0]
diabetes_df = diabetes_df[diabetes_df['SkinThickness'] != 0]
diabetes_df = diabetes_df[diabetes_df['Glucose'] != 0]
diabetes_df = diabetes_df[diabetes_df['BMI'] != 0]
print("Cleaned Shape:", diabetes_df.shape)

# 3. Visualization
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
features_to_plot = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for i, col in enumerate(features_to_plot):
    r, c = divmod(i, 3)
    ax[r, c].scatter(diabetes_df[col], diabetes_df['Outcome'])
    ax[r, c].set_title(col)
plt.tight_layout()
plt.show()

# 4. Train-Test Split
x = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 5. Multi-Model Evaluation
def fMeasure(acc, prec, recall):
    return 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0

def model_results(model):
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    print(f"accuracy_score :  {acc}")
    print(f"precision_score :  {prec}")
    print(f"recall_score :  {recall}")
    print(f"F-measure:  {fMeasure(acc, prec, recall)}")
    print("\n----------------------------\n")

models = {
    "logistic_model": LogisticRegression(penalty='l2', C=1.0, solver='liblinear'),
    "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(solver='svd'),
    "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
    "LinearSVC": LinearSVC(C=1.0, max_iter=1000, tol=1e-2, dual=False),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "GaussianNB": GaussianNB()
}

for name, model in models.items():
    model.fit(x_train, y_train)
    print(f"{name}:")
    model_results(model)

# 6. Hyperparameter Tuning for QDA
param_grid = {
    'reg_param': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'store_covariance': [True, False],
    'tol': [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]
}
grid = GridSearchCV(QuadraticDiscriminantAnalysis(), param_grid, cv=10)
grid.fit(x_train, y_train)

print("Best hyperparameters:", grid.best_params_)
print("Best score:", grid.best_score_)
print("Test accuracy:", grid.best_estimator_.score(x_test, y_test))
```
<H3>Output:</H3>

<img width="1042" height="661" alt="image" src="https://github.com/user-attachments/assets/cee0b50e-21e6-474e-8e0b-868e6e64fdc7" />
<H3>Result:</H3>
Thus, MLP is implemented for multi-classification using python.
