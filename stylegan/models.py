import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import seaborn as sns
import itertools
import operator

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score, \
    recall_score, precision_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics, preprocessing, tree
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_classification
from sklearn.inspection import permutation_importance

# for data and modeling
# import keras
# from keras.callbacks import EarlyStopping
# from keras.wrappers.scikit_learn import KerasClassifier
# from keras.utils import np_utils
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from tensorflow.keras import datasets, layers, models

from six import StringIO
from IPython.display import Image
import pydotplus

from ast import literal_eval

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


def plot_feature_importances(importance):
    importances = pd.DataFrame({'feature': feature_names, 'feature_importance': importance})
    plt.figure(figsize=(12, 10))
    plt.title("Feature importances")
    plt.xlabel("Permutation importance")
    plt.barh(importances["feature"].tolist(), importances["feature_importance"].tolist())
    plt.savefig("nb_heatmap_40.png")


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
    print('Accuracy: {:.5f}'.format(accuracy))

    heatmap_confmat(ytest, y_pred_nb, "naive_bayes.png")

    feature_importance_NB(nb, xtest, ytest)
    print("Naive Bayes done")


def feature_importance_NB(model, xval, yval):
    r = permutation_importance(model, xval, yval, n_repeats=30, random_state=0)
    print(len(r))
    imp = r.importances_mean

    # importance = np.add(imp[40:], imp[:40])
    importance = imp

    # for i in r.importances_mean.argsort()[::-1]:
    #     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
    #         print(f"{feature_names[i]: <8}" f"{r.importances_mean[i]: .3f}" f" +/- {r.importances_std[i]: .3f}")
    plot_feature_importances(importance)
    # importances = pd.DataFrame({'feature': feature_names, 'feature_importance': importance})
    # plt.figure(figsize=(12, 10))
    # plt.title("Feature importances")
    # plt.xlabel("Permutation importance")
    # plt.barh(importances["feature"].tolist(), importances["feature_importance"].tolist())
    # plt.savefig("nb_heatmap_40.png")


def LogRegression(x, y, xtest, ytest):
    # define the model
    model = LogisticRegression()
    # fit the model
    model.fit(x, y)

    # predict y
    ypred = model.predict(xtest)
    print(ypred[:10])
    ypred = [1 if i > 0.6 else 0 for i in ypred]
    accuracy = accuracy_score(ytest, ypred)
    print(accuracy)
    # heatmap_confmat(ytest, ypred, "logregression_heatmap.png")

    imp = np.std(x, 0) * model.coef_[0]
    # imp = model.coef_[0]

    importance = imp
    # importance = np.add(imp[40:], imp[:40])

    feature_importance = pd.DataFrame({'feature': feature_names, 'feature_importance': importance})
    print(feature_importance.sort_values('feature_importance', ascending=False).head(10))

    # plt.barh([x for x in range(len(importance))], importance)
    importances = pd.DataFrame({'feature': feature_names, 'feature_importance': importance})

    plt.figure(figsize=(12, 10))
    plt.title("Feature importances")
    plt.barh(importances["feature"].tolist(), importances["feature_importance"].tolist())
    plt.savefig("logreg_barplot_40.png")

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
    print("Logistic Regression done")


def RandomForest(xtrain, ytrain, xtest, ytest):
    # Create a Gaussian Classifier
    model = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    model.fit(xtrain, ytrain)

    ypred = model.predict(xtest)
    print("Accuracy:", metrics.accuracy_score(ytest, ypred))
    # heatmap_confmat(ytest, ypred, "randomforest_heatmap.png")

    scores = model.feature_importances_
    # scores = np.add(scores[40:], scores[:40])
    print(sorted(zip(map(lambda x: round(x, 4), scores), feature_names),
                 reverse=True))
    importances = pd.DataFrame({'feature': feature_names, 'feature_importance': scores})

    plt.figure(figsize=(12, 10))
    plt.title("Feature importances")
    plt.barh(importances["feature"].tolist(), importances["feature_importance"].tolist())
    plt.savefig("rf_barplot_80_depth100.png")

    # support = model.get_support()
    # print(support)
    # selected_feat = X_train.columns[(sel.get_support())]
    # print(selected_feat)
    # # PermutationImportance(model, xtest, ytest)
    print("random forest done")


def DecisionTree(xtrain, ytrain, xtest, ytest, selection=False):
    if selection:
        feature_names = selection

    model = DecisionTreeClassifier(max_depth=10)

    # Train Decision Tree Classifer
    model = model.fit(xtrain, ytrain)

    # Predict the response for test dataset
    ypred = model.predict(xtest)
    print("Accuracy:", metrics.accuracy_score(ytest, ypred))
    # VisualizationTree(model)
    # heatmap_confmat(ytest, ypred, "decisiontree_heatmap_80.png")
    # print("heatmap saved")
    imp = model.feature_importances_

    # Change for 40 or 80 features:
    # importance = np.add(imp[40:], imp[:40])
    importance = imp

    importances = pd.DataFrame({'feature': feature_names, 'feature_importance': importance})
    print(importances.sort_values('feature_importance', ascending=False).head(10))
    # std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    # PermutationImportance(model, xtest, ytest)

    plt.figure(figsize=(12, 10))
    plt.title("Feature importances")
    plt.barh(importances["feature"].tolist(), importances["feature_importance"].tolist())
    plt.savefig("decisiontree_heatmap_80.png")

    print("decision tree done")


def VisualizationTree(clf):
    feature_cols = [i for i in range(80)]
    target_names = ['0', '1']

    tree.plot_tree(clf,
                   feature_names=feature_cols,
                   class_names=target_names,
                   filled=True,
                   rounded=True)
    plt.figure(figsize=(12, 12))
    plt.savefig('tree_visualization.png', bbox_inches='tight', dpi=100, fontsize=18)


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
    imp = model.named_steps['linearsvc'].coef_
    ypred = model.predict(xtest)
    print("Accuracy:", metrics.accuracy_score(ytest, ypred))
    # heatmap_confmat(ytest, ypred, "svm.png")

    # Uncommend for 80 features
    # scores = np.add(imp[0][40:], imp[0][:40])

    # Uncommend for 40 features
    scores = imp[0]
    # scores = [float(i) / sum(scores) for i in scores]

    sorted_index = sorted(range(len(scores)), key=lambda k: scores[k])
    for i in sorted_index:
        print(str(feature_names[i]) + ": " + str(scores[i]))
    print("SVM done")

    features_names = ['input1', 'input2']
    f_importances(scores, features_names)

    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.savefig("barplot_svm_40.png")

    # importances = pd.DataFrame({'feature': feature_names, 'feature_importance': scores})
    # plt.figure(figsize=(12, 10))
    # plt.title("Feature importances")
    # plt.barh(importances["feature"].tolist(), importances["feature_importance"].tolist())
    # plt.savefig("svm_barplot_40.png")


def Boost(xtrain, ytrain, xtest, ytest):
    # data_dmatrix = xgb.DMatrix(data=xtrain, label=ytrain)
    print(len(xtrain[0]))
    print(len(feature_names))
    x_train = pd.DataFrame(data=xtrain, columns=feature_names)
    x_test = pd.DataFrame(data=xtest, columns=feature_names)

    dtrain = xgb.DMatrix(x_train, label=ytrain)

    model = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                             max_depth=5, alpha=10, n_estimators=10)

    model.fit(x_train, ytrain)
    ypred = model.predict(x_test)
    ypred = [0 if i < 0.5 else 1 for i in ypred]

    print("Accuracy:", metrics.accuracy_score(ytest, ypred))
    # params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
    #             'max_depth': 5, 'alpha': 10}

    # cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
    #                     num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

    # model.plot_tree(xg_reg,num_trees=0)
    # plt.rcParams['figure.figsize'] = [50, 10]
    # plt.savefig("tree_boost.png")

    xgb.plot_importance(model, max_num_features=10)
    plt.rcParams['figure.figsize'] = [20, 20]
    plt.savefig("tree_boost.png")

    # heatmap_confmat(ytest, ypred, "heatmap_boost.png")

    # feature_importance(model, xtest, ypred)

    # rmse = np.sqrt(mean_squared_error(y_test, preds))
    # print("RMSE: %f" % (rmse))


def feature_importance(model, xval, yval):
    importance = model.coef_

    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))

    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.savefig("testfig.png")


def Get_xy(data, one_hot=False, binary=False, type_input="normal"):
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
    # print(train_values, train_counts)
    test_values, test_counts = np.unique(y_test, return_counts=True)
    # print(test_values, test_counts)
    # print(y_train.shape)
    # print(X_train.shape)

    return X_train, y_train, X_test, y_test


def PermutationImportance(model, xtest, ytest):
    r = permutation_importance(model, xtest, ytest, n_repeats=30, random_state=0)
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0.013:
            print(f"{feature_names[i]:<8}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")


pd.set_option('display.expand_frame_repr', False)


def base(feature_names, feature_type, flipped=False):
    if feature_type == "diff":
        DATA_PATH = "/content/drive/MyDrive/ExplainedKinshipData/data/split_diff_features_data.csv"
        data_or = pd.read_csv(DATA_PATH)

        data_or.columns = ["unnamed", "pic1", "pic2", "ptype"] + feature_names

        cols = ['pic1', 'pic2', 'ptype'] + feature_names
        data_orig = data_or[cols]

        if flipped == True:
            data_nonflipped = data_orig.iloc[:, 1:]
            features_split = data_orig.columns[3:].tolist()

            data_flipped = data_orig[["pic2", "pic1", "ptype"] + features_split]
            data_flipped[features_split] = -data_flipped[features_split]
            data_flipped = data_flipped[["pic1", "pic2", "ptype"] + features_split]
            data_flipped.rename(columns={'pic1': 'pic2', 'pic2': 'pic1'}, inplace=True)
            data_nonflipped = data_orig[["pic1", "pic2", "ptype"] + features_split]
            data = pd.concat([data_nonflipped, data_flipped])

        else:
            data = data_orig

        # data = data[cols]
        data["combined"] = (data[data.columns[3:]]).values.tolist()

        # data["combined"] = data[feature_names]
        print(data["combined"])

    elif feature_type == "both":
        DATA_PATH = "/content/drive/MyDrive/ExplainedKinshipData/data/all_pairs_also_unrelated_complete.csv"
        # raw_data = pd.read_csv(DATA_PATH, converters={"feat1": literal_eval, "feat2": literal_eval}, nrows=50)
        raw_data = pd.read_csv(DATA_PATH, converters={"feat1": literal_eval, "feat2": literal_eval})

        data_nonflipped = raw_data[["pic1", "pic2", "ptype", "feat1", "feat2"]]
        if flipped == True:
            data_flipped = raw_data[["pic2", "pic1", "ptype", "feat2", "feat1"]]
            data_flipped.columns = ["pic1", "pic2", "ptype", "feat1", "feat2"]
            data = pd.concat([data_nonflipped, data_flipped])
        else:
            data = data_nonflipped

    ptype_counted = data['ptype'].value_counts()
    print(ptype_counted)

    # Split data in testing and training
    # X_train, y_train, X_test, y_test = Get_xy()
    # X_train, y_train, X_test, y_test = Get_xy(data, binary=True, type_input="normal")
    X_train, y_train, X_test, y_test = Get_xy(data, binary=True, type_input="combined")

    # NeuralNetwork(X_train, y_train, X_test, y_test, feed_forward=True)
    DecisionTree(X_train, y_train, X_test, y_test)
    # NaiveBayes(X_train, y_train, X_test, y_test, binary=True)
    # SVM(X_train, y_train, X_test, y_test)
    # LogRegression(X_train, y_train, X_test, y_test)
    # RandomForest(X_train, y_train, X_test, y_test)
    # Boost(X_train, y_train, X_test, y_test)


feature_names_original = ["Male", "smiling", "attractive", "wavy hair", "young", "5-o-clock-shadow",
                          "arched eyebrows", "bags under eyes", "bald", "bangs", "big lips", "big nose", "black hair",
                          "blond hair", "blurry", "brown hair", "bushyeyebrows", "chubby", "double chin", "eyeglasses",
                          "goatee", "gray hair", "heavy makeup", "highcheekbones", "mouth slightly open", "mustache",
                          "narrow eyes", "no beard", "oval face", "pale skin", "pointy nose", "receding hairline",
                          "rosy cheeks", "sideburns", "straight hair", "wearing earrings", "wearing hat",
                          "wearing lipstick", "wearing necklace", "wearing necktie"]

feature_numbers = [19, 13, 16, 23, 20, 22, 2, 21, 17, 10, 12, 11, 24, 14, 15, 1, 0, 18, 29, 4, 34, 28, 6, 39, 5, 25, 30,
                   31, 32, 35, 37, 27, 36, 26, 38, 3, 33, 8, 7, 9]

# feature_numbers = [11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29,
#                   20, 31, 32, 33, 34, 35, 36, 37, 38, 39, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

feature_numbers2 = [40 + i for i in feature_numbers]
# all_feature_numbers = feature_numbers + feature_numbers2

# Uncommend for just selection of features:
# selection = ["wavy hair", "5-o-clock-shadow",
#              "arched eyebrows", "bags under eyes", "bald", "bangs", "black hair",
#              "blond hair", "brown hair", "bushyeyebrows", "eyeglasses",
#              "gray hair", "highcheekbones",
#              "narrow eyes", "receding hairline",
#              "sideburns", "straight hair", "wearing earrings", "wearing hat"]

feature_names = [feature_names_original[i] for i in feature_numbers]

# feature_names = [feature_names_original[i] for i in feature_numbers] + [feature_names_original[i] + "_2" for i in feature_numbers]
# feature_names = ["feature_" + str(x) for x in feature_names_ordered]

# data = pd.read_csv("/content/drive/MyDrive/ExplainedKinshipData/data/data_topmasked/pic_train_pairs_topmasked.csv")
# print(data)

# DATA_PATH = "/content/drive/MyDrive/ExplainedKinshipData/data/split_diff_features_data.csv"
# data_or = pd.read_csv(DATA_PATH)

# data_or.columns = ["unnamed", "pic1", "pic2", "ptype"] + feature_names

# cols = ['pic1', 'pic2', 'ptype'] + feature_names
# data = data_or[cols]
# corr_mat = data.corr().abs()
# cor = round(corr_mat, 1)
# print(cor)

# sol = (cor.where(np.triu(np.ones(cor.shape), k=1).astype(np.bool))
#                   .stack()
#                   .sort_values(ascending=False))
# print(sol[:20])
# # cor.to_csv("/content/drive/MyDrive/ExplainedKinshipData/data/corrmat.csv")
# # plt = cor.style.background_gradient(cmap='coolwarm')
# plt.subplots(figsize=(20,15))
# sns.heatmap(cor, annot=False)
# plt.savefig("/content/drive/MyDrive/ExplainedKinshipData/data/corrmat.png")

base(feature_names, feature_type="diff")
# base(feature_names, feature_type="both")


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

# ptypes = data["ptype"].tolist()
# first_gen = ["ms", "md", "fs", "fd"]
# zero_gen = ["sibs", "bb", "ss"]
# ptypes_bin = [1 if i in first_gen else 0 if i in second_gen else 2 for i in ptypes]
# print(ptypes_bin)
# tv = data["ptype"].values # Target variable





