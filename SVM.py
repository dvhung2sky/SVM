import numpy as np
from sklearn.svm import SVC  # "Support Vector Classifier"
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2,random_state=0, cluster_std=0.60)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,)


# Linear kernel

model = SVC(kernel='linear')
model.fit(X_train, y_train) 


y_predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, y_predict))



# Polynomial kernel

from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(200, factor=.2, noise=.1)

model = SVC(kernel='poly',degree = 100,C=1000)
model.fit(X, y)


# Gaussion kernel

iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features. We could
y = iris.target
    
model = svm.SVC(kernel='rbf',gamma=0.1)
model.fit(X, y)

