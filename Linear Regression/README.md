# Linear Regression

## Folder Architecture:

- LinearRegression/LinearRegression.py -> This file contains our implementation of Linear Regression. You can import this file and use the class LinearRegression to perform Linear Regression on your data.
- ./test.ipynb -> This file contains the code to test our implementation of Linear Regression. This also compares our implementation with the implementation of Linear Regression in sklearn.

## How to use:

- Import the file LinearRegression.py in your code.
- Create an object of the class LinearRegression.
- Run the fit method on the object to train the model.
- Use the predict method to predict the output for a given input.
- Use the score method to get the R^2 score of the model.

You can also adjust the learning rate and epochs for the model by passing them as arguments while creating the object of the class LinearRegression.

## How to run the test file:
- Open the test.ipynb file in Jupyter Notebook.
- Make sure that sklearn is installed in your system. If not, install it using the command `pip install sklearn`.
- Run the cells in the file to test our implementation of Linear Regression.


## Room for improvement:
- This is not a full implementation of Linear Regression. It only works for a single variable. We can extend this to work for multiple variables.
- We also need to implement different types of gradient descent algorithms like Stochastic Gradient Descent, Mini Batch Gradient Descent, etc.
- We could also implement different types of regularization like L1, L2, etc.
- We could also implement different types of loss functions like Mean Squared Error, Mean Absolute Error, etc.
- We could also implement different types of optimizers like Adam, RMSProp, etc.

