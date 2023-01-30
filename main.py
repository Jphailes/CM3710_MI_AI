import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import plotly.express as px
# import scikit_learn

# Mushroom data file is placed within working directory.
mushrooms = 'agaricus-lepiota.data'

shrooms = pd.read_csv(mushrooms, header=None)

# split the data set between predicted class and attributes.
# the first colum in the dataset is the class, which will e(edible) or (p)poisonous
predicted_class = shrooms[0]
shrooms = shrooms.drop([0], axis=1)

col_labels = ['cap-shape',
              'cap-surface',
              'cap-color',
              'bruises',
              'odor',
              'gill-attachment',
              'gill-spacing',
              'gill-size',
              'gill-color',
              'stalk-shape',
              'stalk-root',
              'stalk-surface-above-ring',
              'stalk-surface-below-ring',
              'stalk-color-above-ring',
              'stalk-color-below-ring',
              'veil-type',
              'veil-color',
              'ring-number',
              'ring-type',
              'spore-print-color',
              'population',
              'habitat'
              ]

# add labels to the dataset to make viewing dataframe more meaningful, but take a copy incase you want the non labelled
# version later
labelled_shrooms = shrooms.copy()
labelled_shrooms.columns = col_labels

print(labelled_shrooms.head(5))
print(predicted_class.head(5))

# One hot encode dem shrooms as all my data points are categorical.
# Single values are stored. Need to use look up to the attribute description file to understand meaning.
# The letters currently represent words. IE cap-colour = n, represents the colour brown.
one_hot_shrooms = pd.get_dummies(labelled_shrooms)

# as my data is now all binary values, then there is not further preprocessing required.
# IE there is no continuous data so no need to scale or normalise, etc.
print(one_hot_shrooms.head(5))

# convert to numpy array to improve performance
x = one_hot_shrooms.to_numpy()
y = predicted_class.to_numpy()

"""
First lets try a KNN algorithm. KNN is a lazy algorithm which means it does not need any training points for model generation.
We will try different values of K and different weightings. 
"""

# Create a KNeighborsClassifier object and set number of neighbours to 5 initially.
knn_model5 = KNeighborsClassifier(n_neighbors=5, weights='distance') # try distance and uniform

## create a test train split leaving the split size at 20% and then check the sizes of the respective sets.
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#
# print(f'x_train dimensions = {x_train.shape} and x_test dimensions = {x_test.shape}')
# print(f'y_train dimensions = {y_train.shape} and y_test dimensions = {y_test.shape}')
"""
Rather than manually create a train test split, use a stratifiedkfold.
"""
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(x, y)
print(skf)

# if we want to view how each fold is split between train and test.....
for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")

# Now use the KNN model we have created and the skf train/test split
# This will fit the data to the KNN model 5 times w.r.t skf
scores5 = cross_val_score(knn_model5, x, y, cv=skf)

# We can see the scores created for each of the 5 folds
def print_score(scores):
    for i in range(len(scores)):
        print(f'Fold {i} score: {scores[i]}')

print_score(scores5)

# and then calculate the mean and the standard deviation of the scores.
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores5.mean(), scores5.std()))


# try some different values of neighbours
print('Using 3 neighbours:')
knn_model3 = KNeighborsClassifier(n_neighbors=3, weights='distance') # using 'distance' improves accuracy slightly
scores3 = cross_val_score(knn_model3, x, y, cv=skf)
print_score(scores3)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores3.mean(), scores3.std()))
print('\n')

print('Using 2 neighbours:')
knn_model2 = KNeighborsClassifier(n_neighbors=2, weights='distance', p=1)  # p1= euclidean, p2= manhattan (using minkowksi metric)
scores2 = cross_val_score(knn_model2, x, y, cv=skf)
print_score(scores2)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores2.mean(), scores2.std()))

"""So the model appears to improve with less neighbours. I would be inclined to leave it at 2 neighbours with the accuracy
increasing to 95% and the standard deviation reducing to 0.06. I did test going down to 1 but there was no change from 2.
Also tried changing the distance metric from Euclidean (P=1) and Manhattan (P=2) but this did not make any difference."""


# show the parameters used in the final model
knn_model2.get_params()

"""
#######################################################################################################################
Now because our problem is a non linear binary classification problem the 2nd algorithm i will look to implement will be
a decision tree. It also makes sense in my opinion, as the data we have consists of attributes which are categorical 
descriptors of a property of the mushroom. (nom nom).
 
As the decision tree can be visualised it may also help to identify certain characteristics of a mushroom that may make
it poisonous or edible
#######################################################################################################################
"""

# our data does not need to be one hot encoded for decision tree classification so will recreate the train test split
# but ahhhh.... SKLEARN currently does not handle categorical data !! However my data is not ordinal so i can still use
# the previously one hot encoded data.

from sklearn import tree
from sklearn.metrics import accuracy_score

# i have initially gone with a depth of 10 here as dateset has 22 attributes.
tree_model = tree.DecisionTreeClassifier(max_depth=10)

#This time i will use a simple manual split of the data
# create a test train split leaving the split size at 30% and then check the sizes of the respective sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

print(f'x_train dimensions = {x_train.shape} and x_test dimensions = {x_test.shape}')
print(f'y_train dimensions = {y_train.shape} and y_test dimensions = {y_test.shape}')

# fit the model
tree_model.fit(x_train, y_train)

# create predictions suing the test split
tree_predict = tree_model.predict(x_test)

acc = accuracy_score(y_test, tree_predict)

# Here we can see the accuracy. Which appears to be pretty good.
print(acc)

#So lets look at the model parameters
tree_model.get_params()
# and check the depth of the tree. remembering i initially set this to 10 when instantiating the classifier
tree_model.get_depth()

# Note it has a depth of 7 so i am going to manually check the depth to use:
def mytree(max_depth, criteron):
    for i in range(1, max_depth):
        tree_model = tree.DecisionTreeClassifier(max_depth=i, criterion=criteron)
        tree_model.fit(x_train, y_train)
        tree_predict = tree_model.predict(x_test)
        acc = accuracy_score(y_test, tree_predict)
        print(f'max_depth of {i}, returns accuracy of: {acc}')

# we can see below how the depth of teh tree impacts the accuracy as the depth increases.
mytree(10,criteron='entropy')

tree_model = tree.DecisionTreeClassifier(max_depth=7,criterion='entropy')
tree_model.fit(x_train, y_train)
tree_model.get_params()
tree_model.get_depth()
tree_model.get_n_leaves()

tree_model.predict_log_proba(x_test)