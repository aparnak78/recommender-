# recommender-
Here we build a recommender system, using recurrent neural networks that recommender suitable courses for the user from the set of available courses based on their previous knowledge.

Implementation:

Data:

+ We took the dataset from Open University Learning Analytics Dataset (OULAD) 
+ contains data about courses, students and their interaction with virtual learning environment (VLE) for twenty two selected courses (calledmodules).
+ The dataset consists of a table stored in csv format. The table consists of 32594 entries of marks obtained by 28785
students for the 22 subjects.

Model architechture:

+ The models are build using Python 3.7 with Keras framework for deep learning.
+ After the data is loaded and pre-processed the data is split into train set and test set.
+ The train data is used to train the model.
+ The trained model is then evaluated by applying the model to the test data.
+ Now the model is used to predict output for the new inputs.
+ These predicted values are used to make the required recommendations.

+We build three different models- a NN model, a LSTM model and
a GRU model. After importing the data the layers are defined for each model.

NN Model:
+In the Neural network model there are two nodes in the input layer whose output is passed to a
concatenation node in the next layer which concatenates both the inputs to one vector. 
+The output of this concatenation layer is given to the dense layers with 32 nodes and sigmoid function
as activation function.
+ The output is taken from an output layer with one node.

LSTM Model:

+ In the LSTM model the two input arrays are joined to form a single array (three dimensional) hence only on
node is present in the input layer
+ whose output is given to a LSTM layer which has relu activation function.
+ The output from the LSTM layer is then given to a dense layer.
+ The output from the dense layer goes to an output layer with a single node.

GRU Model:
+ In the GRU model the layer are same as that of the LSTM model except for the LSTM layer which is replaced with LSTM.

+ After defining the layers the models are compiled.
+ each model is executed on the training set after compiling. The model is trained on the selected data 
using the fit() function. 
+ After training the model on the training set the output of the model is tested using the test set. This is done by
calling the evaluate() function on the test set. 
+ Now prediction is done using the predict() function.
+ Then recommendations are made using the predicted values for each module.



RESULT:
NN model
Accuracy: 0.713750
Precision: 0.532100
Recall: 0.654200
F1 score: 0.582100

LSTM model:
Accuracy: 0.892000
Precision: 0.521000
Recall: 0.674000
F1 score: 0.583000

GRU model:
Accuracy: 0.923000
Precision: 0.742100
Recall: 0.713400
F1 score: 0.721300
