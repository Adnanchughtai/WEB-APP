# Import libraries
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# App heading
st.write('''
# Explore different ML models and datasets
Will see that which model is good for which dataset?
''')

# Dataset selection and sidebar
dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Breast Cancer', 'Wine Dataset'))

# Model selection and sidebar
classifier_name = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'Random Forest'))

# Function to get dataset
def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

# Function to add parameters
X, y = get_dataset(dataset_name)

# Function to get classifier
st.write('shape of dataset:', X.shape)
st.write('shape of classes:', len(np.unique(y)))

# different parameters for different classifiers
def add_parameter_ui(classifier_name): # ui (user input)
    params = dict() # Create an empty dictionary
    if classifier_name == 'SVM':
        c = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = c # its the degree of correct classifier
    elif classifier_name == 'KNN':
        k = st.sidebar.slider('K', 1, 15)
        params['K'] = k  # Its the number of neighbors
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth # Its the maximum depth of the tree
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators # Its the number of trees
    return params
# Call the function
params = add_parameter_ui(classifier_name)

# Now we will make classifier based on calssifier name and params
def get_classifier(classifier_name, params):
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1234)
    return clf

# Now we will call this function
clf = get_classifier(classifier_name, params)

# To split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Now we will train the classifier
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Check model accuracy
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)

# PLOT DATASET
# Plot dataset
pca = PCA(2)
X_projected = pca.fit_transform(X)

# now we will slice in 0 and 1 dimensions to get
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, 
            c=y, alpha=0.8,
            cmap = 'viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

# plt.show()
st.pyplot(fig)
        