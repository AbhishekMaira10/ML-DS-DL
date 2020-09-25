# -*- coding: utf-8 -*-
"""walmart_trip_type_classification.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1ZCzfKbxU5OqjtBNS85xdZSpnM3ugai2W
"""

# Commented out IPython magic to ensure Python compatibility.
#EDA and Preprocessing
import numpy as np
import pandas as pd
from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# %config InlineBackend.figure_formats = {'png', 'retina'}

train = pd.read_csv("/content/train.csv")
test = pd.read_csv("/content/test.csv")
print(train.shape)
train.tail()

train.info()

plt.figure(figsize=(7, 5))
train.isnull().sum().plot(kind='bar')
plt.xticks(rotation=45, ha='right')
plt.show()

wd = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
      "Friday": 4, "Saturday": 5, "Sunday": 6}
train["Weekday"] = train["Weekday"].apply(lambda x: wd[x])
test["Weekday"] = test["Weekday"].apply(lambda x: wd[x])

def float_to_str(obj):
    """
    Convert Upc code from float to string
    Use this function by applying lambda
    :param obj: "Upc" column of DataFrame
    :return: string converted Upc removing dot.
    """
    while obj != "nan":
        obj = str(obj).split(".")[0]
        return obj

train["Upc"] = train.Upc.apply(float_to_str)
test["Upc"] = test.Upc.apply(float_to_str)

train['TripType'].value_counts()
plt.figure(figsize = (12, 10))

sns.set_style('whitegrid')
ax = sns.countplot(x = 'TripType', data = train, palette = 'mako')
ax.set(title = 'The Frequent of Trip Type', ylabel = 'Counts', xlabel = 'Trip Type')

train_dd = pd.get_dummies(train["DepartmentDescription"])
test_dd = pd.get_dummies(test["DepartmentDescription"])

train_dd = pd.concat([train[["VisitNumber"]], train_dd], axis=1)
test_dd = pd.concat([test[["VisitNumber"]], test_dd], axis=1)

train_dd = train_dd.groupby("VisitNumber", as_index=False).sum()
test_dd = test_dd.groupby("VisitNumber", as_index=False).sum()
train_dd.tail()

train = train.merge(train_dd, on=["VisitNumber"])

#remove null and infinite values
train.replace(np.inf, 0, inplace=True)
train.fillna(value=0, inplace=True)

test.replace(np.inf, 0, inplace=True)
test.fillna(value=0, inplace=True)
train.drop("DepartmentDescription", axis=1, inplace=True)
test.drop("DepartmentDescription", axis=1, inplace=True)
train.drop("VisitNumber", axis=1, inplace=True)
test.drop("VisitNumber", axis=1, inplace=True)

train.loc[train["ScanCount"] < 0, "Return"] = 1
train.loc[train["Return"] != 1, "Return"] = 0

test.loc[test["ScanCount"] < 0, "Return"] = 1
test.loc[test["Return"] != 1, "Return"] = 0

train["Pos_Sum"] = train["ScanCount"]
test["Pos_Sum"] = test["ScanCount"]

train.loc[train["Pos_Sum"] < 0, "Pos_Sum"] = 0
test.loc[test["Pos_Sum"] < 0, "Pos_Sum"] = 0

train["Neg_Sum"] = train["ScanCount"]
test["Neg_Sum"] = test["ScanCount"]

train.loc[train["Neg_Sum"] > 0, "Neg_Sum"] = 0
test.loc[test["Neg_Sum"] > 0, "Neg_Sum"] = 0

train.drop("ScanCount", axis=1, inplace=True)
test.drop("ScanCount", axis=1, inplace=True)
# upc is unique for each customer i.e no need to put it in model training
train.drop("Upc", axis=1, inplace=True)
test.drop("Upc", axis=1, inplace=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train[['Weekday']])
var = scaler.transform(train[['Weekday']])

train['Weekday'] = var
scaler1 = MinMaxScaler()
scaler1.fit(train[['FinelineNumber']])
var1 = scaler1.transform(train[['FinelineNumber']])
train['FinelineNumber'] = var1

train.head()
print(train.shape)

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder

classifier = keras.Sequential()
classifier.add(tf.keras.layers.Dense( 73 , activation = tf.nn.relu))
classifier.add(tf.keras.layers.Dense( 5 , activation = tf.nn.relu))
classifier.add(tf.keras.layers.Dense( 5 , activation = tf.nn.relu))
classifier.add(tf.keras.layers.Dense(38 , activation = tf.nn.softmax))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
metrics = ['accuracy'])
from sklearn.model_selection import train_test_split


y =  train['TripType']


label_enc = LabelEncoder().fit(y)
y_label = label_enc.transform(y)
y_cat = to_categorical(y_label)

x = train.drop('TripType',1)


x_train, x_test, y_train, y_test = train_test_split(
        x, y_cat, test_size=0.33, random_state=0) 
classifier.fit(x_train, y_train, epochs = 50)

history = classifier.fit(x_train, y_train, validation_data=(x_test, y_test),epochs=20)

from sklearn.metrics import accuracy_score
y_pred= classifier.predict(x_test)

ans1 = []
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for i in range(137094):
  sub_val = np.argmax(y_test[i])
  ans1.append(sub_val)
print(ans1)
le.fit(ans1)
ans2 = []
for i in range(137094):
  sub_val = np.argmax(y_pred[i])
  ans2.append(sub_val)
final_ans = le.inverse_transform(ans2)
df = pd.DataFrame(final_ans)
df.to_csv('Abhishek_101703016.csv', index = False)



l = classifier.predict(x_test)
col = []
for i in sorted(np.unique(train.TripType)):
  col.append('TripType_'+str(i))
def change(s):
  return np.round(s,0)
sub = pd.DataFrame(data = l, columns=col)
for i in col:
  sub[i] = sub[i].apply(change)

sub['VisitNumber'] = test['VisitNumber']
sub.to_csv('final.csv', index = False, header = True)
sub.drop_duplicates(keep="first",inplace=True)
sub.to_csv('Abhishek_101703026.csv', index = False, header = True)