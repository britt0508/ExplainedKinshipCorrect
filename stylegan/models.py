import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
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

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# for data and modeling
import keras
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.keras import datasets, layers, models

from six import StringIO
from IPython.display import Image
import pydotplus

from ast import literal_eval

from collections import Counter

DATA_PATH = "/content/drive/MyDrive/ExplainedKinshipData/data/all_pairs_also_unrelated_2.csv"
data = pd.read_csv(DATA_PATH, converters={"feat1": literal_eval, "feat2": literal_eval})
print(data["ptype"])
# data = data.iloc[-101830:]
ptype_counted = data['ptype'].value_counts()
print(ptype_counted)

def NaiveBayes(x, y, xtest, ytest, binary=False):
    if binary:
        nb = GaussianNB()
    else:
        nb = CategoricalNB()
    nb.fit(x, y)
    nb.predict(xtest)
    y_pred_nb = nb.predict(X_test)
    y_prob_pred_nb = nb.predict_proba(X_test)
    # how did our model perform?
    count_misclassified = (y_test != y_pred_nb).sum()

    print("CategoricalNB")
    print("=" * 30)
    print('Misclassified samples: {}'.format(count_misclassified))
    accuracy = accuracy_score(y_test, y_pred_nb)
    print('Accuracy: {:.2f}'.format(accuracy))

    # print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))


def LinRegression(x, y, xtest, ytest):
    model = LinearRegression().fit(x, y)
    r_sq = model.score(xtest, ytest)
    y_pred = model.predict(xtest)
    print(r_sq)


def DecisionTree(x, y, xtest, ytest):
    clf = DecisionTreeClassifier(max_depth=5)

    # Train Decision Tree Classifer
    clf = clf.fit(x, y)

    # Predict the response for test dataset
    y_pred = clf.predict(xtest)
    print("Accuracy:", metrics.accuracy_score(ytest, y_pred))
    VisualizationTree(clf)


def VisualizationTree(clf):
    feature_cols = [i for i in range(80)]
    # feature_names = df.columns[:5]
    target_names = data['ptype'].unique().tolist()

    tree.plot_tree(clf,
                   feature_names=feature_cols,
                   class_names=target_names,
                   filled=True,
                   rounded=True)

    plt.savefig('tree_visualization.pdf')
    # dot_data = StringIO()
    # export_graphviz(clf, out_file=dot_data,
    #                 filled=True, rounded=True,
    #                 special_characters=True,feature_names = feature_cols,class_names=['0','1'])
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_png('diabetes.png')
    # Image(graph.create_png())


def NeuralNetwork(X, y, xtest, ytest, feed_forward=False):
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    model = Sequential()
    if feed_forward:
        model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
        model.add(Dense(128, activation="sigmoid"))
        model.add(Dense(10, activation="softmax"))
        model.summary()
    else:
        model.add(Dense(16, input_shape=(X.shape[1],), activation='relu'))  # input shape is (features,)
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(Dense(11, activation='softmax'))
        model.summary()

    # compile the model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  # this is different instead of binary_crossentropy (for regular classification)
                  metrics=['accuracy'])

    # early stopping callback
    # This callback will stop the training when there is no improvement in
    # the validation loss for 10 consecutive epochs.
    es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                       mode='min',
                                       patience=10,
                                       restore_best_weights=True)  # important - otherwise you just return the last weigths...

    # now we just update our model fit call
    history = model.fit(X,
                        dummy_y,
                        callbacks=[es],
                        epochs=8000000,  # you can set this to a big number!
                        batch_size=10,
                        shuffle=True,
                        validation_split=0.2,
                        verbose=1)

    history_dict = history.history

    # learning curve
    # accuracy
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']

    # loss
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    # range of X (no. of epochs)
    epochs = range(1, len(acc) + 1)

    # plot
    # "r" is for "solid red line"
    plt.plot(epochs, acc, 'r', label='Training accuracy')
    # b is for "solid blue line"
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    preds = model.predict(X)  # see how the model did!
    print(preds[0])  # i'm spreading that prediction across three nodes and they sum to 1
    print(np.sum(preds[0]))  # sum it up! Should be 1

    # Almost a perfect prediction
    # actual is left, predicted is top
    # names can be found by inspecting Y
    matrix = confusion_matrix(dummy_y.argmax(axis=1), preds.argmax(axis=1))
    print(matrix)

    # more detail on how well things were predicted
    print(classification_report(dummy_y.argmax(axis=1), preds.argmax(axis=1)))


def SVM(X, y, xtest, ytest):
    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, multi_class="crammer_singer"))
    clf.fit(X, y)
    # print(clf.named_steps['linearsvc'].coef_)
    # print(clf.named_steps['linearsvc'].intercept_)
    ypred = clf.predict(xtest)
    print(confusion_matrix(ytest, ypred))
    print(classification_report(ytest, ypred))


def Get_xy(one_hot=False, binary=False):
    data["feat1and2"] = data["feat1"] + data["feat2"]
    f = data["feat1and2"]

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

    if binary:
        X_train = np.array(list(X_train))
        X_test = np.array(list(X_test))
    else:
        # DO SOMETHING HERE WITH DATA TO MAKE IT WORK
        X_train = np.array(list(map(list, X_train)))
        # print(X_train)
        y_train = np.squeeze(np.array(list(y_train)))

        X_test = np.array(list(map(list, X_test)))
        # print(X_test)
        y_test = np.squeeze(np.array(list(y_test)))

    print(y_test)
    train_values, train_counts = np.unique(y_train, return_counts=True)
    print(train_values, train_counts)
    test_values, test_counts = np.unique(y_test, return_counts=True)
    print(test_values, test_counts)
    # print(y_train.shape)
    # print(X_train.shape)

    return X_train, y_train, X_test, y_test


# X_train, y_train, X_test, y_test = Get_xy()
# X_train, y_train, X_test, y_test = Get_xy(one_hot = True)
X_train, y_train, X_test, y_test = Get_xy(binary = True)

# NeuralNetwork(X_train, y_train, X_test, y_test, feed_forward=True)
# DecisionTree(X_train, y_train, X_test, y_test)
NaiveBayes(X_train, y_train, X_test, y_test, binary=True)
# SVM(X_train, y_train, X_test, y_test)


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





