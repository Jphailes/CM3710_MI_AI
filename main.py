import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# Mushroom data file is placed within working directory.
mushrooms = 'agaricus-lepiota.data'

shrooms = pd.read_csv(mushrooms, header=None)

#split the data set between predicted class and attributes.
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

# add labels to the dataset to make viewing dataframe more meaningful
labelled_shrooms = shrooms.copy()
labelled_shrooms.columns = col_labels

print(labelled_shrooms.head())
print(predicted_class)

# one hot encode dem shrooms
one_hot_shrooms = pd.get_dummies(labelled_shrooms)

# convert to numpy to improve performance
x = one_hot_shrooms.to_numpy()
y = predicted_class.to_numpy()


# Lets fit our model:




