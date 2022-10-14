## K-Nearest Neighbors Classifier Algorithm

This is a simple implementation of the K-Nearest Neighbors Classifier Algorithm. The algorithm is implemented in Python 3.6. The algorithm is implemented from scratch without using any machine learning libraries.

### Usage

The algorithm is implemented in the file `knn.py`. The file `knn.py` contains the class `KNN` which is the implementation of the K-Nearest Neighbors Classifier Algorithm. The class `KNN` has the following methods:

* `__init__(self, k=3, distance='euclidean')` - The constructor of the class. The parameter `k` is the number of nearest neighbors to consider. The parameter `distance` is the distance metric to use. The default value of `k` is 3 and the default value of `distance` is 'euclidean'. The distance metric can be either 'euclidean' or 'manhattan'.
* `fit(self, X, y)` - The method to train the model. The parameter `X` is the training data and the parameter `y` is the training labels. 
* `predict(self, X)` - The method to predict the labels of the data. The parameter `X` is the data to predict the labels for. The method returns the predicted labels. 
* `score(self, X, y)` - The method to calculate the accuracy of the model. The parameter `X` is the data to predict the labels for and the parameter `y` is the true labels. The method returns the accuracy of the model.


### Example

The file `test.py` contains an example of how to use the class `KNN`. The example uses the Iris dataset to train the model and test the model. The example uses the following code to train the model:
```
    knn = KNN(k=3, distance='euclidean')
    knn.fit(X_train, y_train)
```
The example uses the following code to test the model:
```
    y_pred = knn.predict(X_test)
    accuracy = knn.score(X_test, y_test)
```

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

### Room for Improvement

* Only Brute Force method is implemented. Other methods like KD-Tree and Ball Tree can be implemented.
* Documentation can be improved.
* The code can be optimized.

### Contributions and Feedback

Contributions and feedback are welcome. Please open an issue or a pull request.
