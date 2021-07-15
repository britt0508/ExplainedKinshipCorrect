import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow as tf

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, \
    recall_score, precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn import preprocessing
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.inspection import permutation_importance

# for data and modeling
import keras
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.keras import datasets, layers, models

import seaborn as sns

from six import StringIO
from IPython.display import Image
import pydotplus

from ast import literal_eval

from sklearn.inspection import permutation_importance

from collections import Counter


def heatmap_confmat(ytest, ypred, name):
    labels = [0, 1]
    conf_mat = confusion_matrix(ytest, ypred, labels=labels)
    print(conf_mat)
    # heatm = sns.heatmap(conf_mat, annot=True)
    # print(heatm)
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in conf_mat.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in conf_mat.flatten() / np.sum(conf_mat)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    heat = sns.heatmap(conf_mat, annot=labels, fmt='', cmap='Blues')
    heat.figure.savefig(name)


def NaiveBayes(xtrain, ytrain, xtest, ytest, binary=False):
    if binary:
        nb = GaussianNB()
        model = "GaussianNB"
    else:
        nb = CategoricalNB()
        model = "CategoricalNB"

    nb.fit(xtrain, ytrain)
    nb.predict(xtest)
    y_pred_nb = nb.predict(xtest)
    y_prob_pred_nb = nb.predict_proba(xtest)
    # how did our model perform?
    count_misclassified = (ytest != y_pred_nb).sum()

    print(model)
    print("=" * 30)
    print('Misclassified samples: {}'.format(count_misclassified))
    accuracy = accuracy_score(ytest, y_pred_nb)
    print('Accuracy: {:.2f}'.format(accuracy))

    heatmap_confmat(ytest, y_pred_nb, "naive_bayes.png")

    feature_importance_NB(nb, xtest, ytest)


def feature_importance_NB(model, xval, yval):
    r = permutation_importance(model, xval, yval, n_repeats=30, random_state=0)
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{feature_names[i]: <8}" f"{r.importances_mean[i]: .3f}" f" +/- {r.importances_std[i]: .3f}")


def LinRegression(x, y, xtest, ytest):
    # define the model
    model = LogisticRegression()
    # fit the model
    model.fit(x, y)

    # predict y
    ypred = model.predict(xtest)
    ypred = [1 if i > 0.5 else 0 for i in ypred]
    accuracy = accuracy_score(ytest, ypred)
    print(accuracy)
    print("lin/log regression done")

    feature_importance = pd.DataFrame({'feature': feature_names, 'feature_importance': model.coef_[0]})
    print(feature_importance.sort_values('feature_importance', ascending=False).head(10))

    # xpos = [x for x in range(len(importance))]
    # plt.bar(xpos, importance)
    # plt.xticks(xpos, feature_names_trimmed)
    # plt.savefig("linreg.png")
    # w = model.coef_[0]
    # feature_importance = pd.DataFrame(feature_names, columns = ["feature"])
    # feature_importance["importance"] = pow(math.e, w)
    # feature_importance = feature_importance.sort_values(by = ["importance"], ascending=False)

    # ax = feature_importance.plot.barh(x='feature', y='importance')
    # plt.savefig("linreg.png")


def RandomForest(xtrain, ytrain, xtest, ytest):
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(xtrain, ytrain)

    ypred = clf.predict(xtest)
    print("Accuracy:", metrics.accuracy_score(ytest, ypred))
    heatmap_confmat(ytest, ypred, "random_forest.png")
    print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), feature_names),
                 reverse=True))
    print("random forest done")


def DecisionTree(xtrain, ytrain, xtest, ytest):
    model = DecisionTreeClassifier(max_depth=4)

    # Train Decision Tree Classifer
    model = model.fit(xtrain, ytrain)

    # Predict the response for test dataset
    ypred = model.predict(xtest)
    print("Accuracy:", metrics.accuracy_score(ytest, ypred))
    # VisualizationTree(model)
    heatmap_confmat(ytest, ypred, "Decisiontree.png")

    print("decision tree done")

    importances = pd.DataFrame({'feature': feature_names, 'feature_importance': model.feature_importances_})
    print(importances.sort_values('feature_importance', ascending=False).head(10))
    # std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)


def VisualizationTree(clf):
    feature_cols = [i for i in range(80)]
    # feature_names = df.columns[:5]
    # target_names = data['ptype'].unique().tolist()
    target_names = ['0', '1']

    tree.plot_tree(clf,
                   feature_names=feature_cols,
                   class_names=target_names,
                   filled=True,
                   rounded=True)
    plt.figure(figsize=(12, 12))
    plt.savefig('tree_visualization.png', bbox_inches='tight', dpi=100, fontsize=18)
    # dot_data = StringIO()
    # export_graphviz(clf, out_file=dot_data,
    #                 filled=True, rounded=True,
    #                 special_characters=True,feature_names = feature_cols,class_names=['0','1'])
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_png('diabetes.png')
    # Image(graph.create_png())


def NeuralNetwork(xtrain, ytrain, xtest, ytest, feed_forward=False):
    print('X_train:', np.shape(xtrain))
    print('y_train:', np.shape(ytrain))
    print('X_test:', np.shape(xtest))
    print('y_test:', np.shape(ytest))

    model = Sequential()
    # if feed_forward:
    model.add(Dense(256, input_shape=(287399, 80), activation="sigmoid"))
    model.add(Dense(128, activation="sigmoid"))
    model.add(Dense(10, activation="softmax"))
    model.add(Dense(1, activation='hard_sigmoid'))
    model.summary()

    sgd = keras.optimizers.SGD(lr=0.5, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # early stopping callback
    # This callback will stop the training when there is no improvement in
    # the validation loss for 10 consecutive epochs.
    es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                       mode='min',
                                       patience=10,
                                       restore_best_weights=True)  # important - otherwise you just return the last weigths...

    # train_data = tf.data.Dataset.from_tensor_slices(xtrain, ytrain)
    model.fit(xtrain, ytrain, epochs=30)
    ypred = model.predict(xtest)
    ypred = [1 if i > 0.5 else 0 for i in ypred]

    loss_and_metrics = model.evaluate(xtest, ytest)
    print('Loss = ', loss_and_metrics[0])
    print('Accuracy = ', loss_and_metrics[1])
    heatmap_confmat(ytest, ypred, "neuralnet.png")
    print("neural network done")


def SVM(xtrain, ytrain, xtest, ytest):
    model = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, multi_class="crammer_singer"))
    model.fit(xtrain, ytrain)
    # print(clf.named_steps['linearsvc'].coef_)
    # print(clf.named_steps['linearsvc'].intercept_)
    ypred = model.predict(xtest)
    print("Accuracy:", metrics.accuracy_score(ytest, ypred))
    heatmap_confmat(ytest, ypred, "svm.png")

    print(classification_report(ytest, ypred))
    print("svm done")

    coef = model.coef_.ravel()
    print(coef)


def feature_importance(model, xval, yval):
    importance = model.coef_

    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))

    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.savefig("testfig.png")


def Get_xy(one_hot=False, binary=False, type_input="normal"):
    # data = data.iloc[-101830:] # Uncommend for balanced dataset when using binary
    # data_split = pd.read_csv("/content/drive/MyDrive/ExplainedKinshipData/data/split_features_data.csv")

    if type_input == "combined":
        f = data["combined"]
    elif type_input == "normal":
        data["feat1and2"] = data["feat1"] + data["feat2"]
        f = data["feat1and2"]
    else:
        f = data["tuples"]

    classes = data["ptype"].values

    labels = ["sibs", "bb", "ss", "ms", "md", "fs", "fd", "gfgd", "gfgs", "gmgd", "gmgs"]

    if binary:
        classes = [1 if i in labels else 0 for i in classes]

    if one_hot:
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        classes = le.transform(data["ptype"])

    X_train, X_test, y_train, y_test = train_test_split(f, classes, test_size=0.3, random_state=42)

    print("Data split")
    print(X_train)

    if binary:
        X_train = list(X_train)
        X_test = list(X_test)
        # print(X_train)
    else:
        X_train = np.array(list(map(list, X_train)))
        y_train = np.squeeze(np.array(list(y_train)))

        X_test = np.array(list(map(list, X_test)))
        y_test = np.squeeze(np.array(list(y_test)))

    # print(y_test)
    train_values, train_counts = np.unique(y_train, return_counts=True)
    print(train_values, train_counts)
    test_values, test_counts = np.unique(y_test, return_counts=True)
    print(test_values, test_counts)
    # print(y_train.shape)
    # print(X_train.shape)

    return X_train, y_train, X_test, y_test


pd.set_option('display.expand_frame_repr', False)

# both features
DATA_PATH = "/content/drive/MyDrive/ExplainedKinshipData/data/all_pairs_also_unrelated_complete.csv"
data = pd.read_csv(DATA_PATH, converters={"feat1": literal_eval, "feat2": literal_eval})
ptype_counted = data['ptype'].value_counts()
print(ptype_counted)

feature_numbers = [19, 13, 16, 23, 20, 22, 2, 21, 17, 10, 12, 11, 24, 14, 15, 1, 0, 18, 29, 4, 34, 28, 6, 39, 5, 25, 30,
                   31, 32, 35, 37, 27, 36, 26, 38, 3, 33, 8, 7, 9]
feature_numbers2 = [40 + i for i in feature_numbers]
all_feature_numbers = feature_numbers + feature_numbers2
feature_names = ["feature_" + str(x) for x in all_feature_numbers]

print(feature_names)

# Difference features
# DATA_PATH = "/content/drive/MyDrive/ExplainedKinshipData/data/split_diff_features_data.csv"
# data = pd.read_csv(DATA_PATH)
# print(data)

# for i in range(40):
#     data["tuples" + str(i)] = list(zip(data.iloc[:, i+4], data.iloc[:, i+44]))
# # list3 = [(list1[i], list2[i]) for i in range(40)]
# data["tuples"] = data[data.columns[44:]].values.tolist()
# print(data)

# uncomment for abs values
# data[["feature_" + str(i) for i in range(40)]] = data[["feature_" + str(i) for i in range(40)]].abs()
# cols = ["pic1", "pic2", "ptype"] + ["feature_" + str(i) for i in range(40)]
# data = data[cols]
# data["combined"] = (data[data.columns[4:]]).values.tolist()
# print(data["combined"])

# ptype_counted = data['ptype'].value_counts()
# print(ptype_counted)
# feature_names = ["feature_" + str(i) for i in range(40)]
# print(feature_names)


# X_train, y_train, X_test, y_test = Get_xy()
# X_train, y_train, X_test, y_test = Get_xy(one_hot = True)
# X_train, y_train, X_test, y_test = Get_xy(binary=True, type_input="combined")
# X_train, y_train, X_test, y_test = Get_xy(binary=True, type_input="tuples")
X_train, y_train, X_test, y_test = Get_xy(binary=True, type_input="normal")

# NeuralNetwork(X_train, y_train, X_test, y_test, feed_forward=True)
DecisionTree(X_train, y_train, X_test, y_test)
# NaiveBayes(X_train, y_train, X_test, y_test, binary=True)
# SVM(X_train, y_train, X_test, y_test)
# LinRegression(X_train, y_train, X_test, y_test)
# RandomForest(X_train, y_train, X_test, y_test)


# features = data[["feat1", "feat2"]]
# print(x)
# features = x.extend(y)


# DATA_PATH_TEST = "/content/drive/MyDrive/ExplainedKinshipData/data/merged.csv"
# # LinRegression(features, ptypes)
# test_data = pd.read_csv(DATA_PATH_TEST)
# x_test = test_data["feature_1"].tolist()
# y_test = test_data["feature_2"].tolist()
# features_test = x_test.extend(y_test)
# ptypes_test = test_data["ptype"].tolist()

# Treee(features, ptypes, features_test, ptypes_test)

# data["feat1"] = data.feat1.apply(literal_eval)
# data["feat2"] = data.feat2.apply(literal_eval)

# f1 = data["feat1"].tolist()
# f2 = data["feat2"].tolist()
# print(f1)
# print(f2)

# f = data[["feat1", "feat2"]].values
# print(f)
# f = data.drop(["pic1", "pic2", "p1", "p2", "ptype"], axis=1).values
# print(f)
# data["feat1and2"] = data["feat1"] + data["feat2"]
# f = data["feat1and2"]

# le = preprocessing.LabelEncoder()
# le.fit(["ms", "md", "fs", "fd", "sibs", "bb", "ss", "gfgd", "gfgs", "gmgd", "gmgs"])
# values = le.transform(data["ptype"])
# X_train, X_test, y_train, y_test = train_test_split(f, values, test_size=0.3, random_state=1)

# Get one hot encoding of columns ptype
# one_hot = pd.get_dummies(data['ptype'])
# one_hot['combined']= one_hot.values.tolist()
# values = one_hot["combined"].values

# ptypes = data["ptype"].tolist()
# first_gen = ["ms", "md", "fs", "fd"]
# zero_gen = ["sibs", "bb", "ss"]
# ptypes_bin = [1 if i in first_gen else 0 if i in second_gen else 2 for i in ptypes]
# print(ptypes_bin)
# tv = data["ptype"].values # Target variable





