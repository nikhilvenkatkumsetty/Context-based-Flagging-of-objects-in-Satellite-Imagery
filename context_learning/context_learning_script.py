import numpy as np
import generate_features
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import preprocessing

# Standard sklearn classifiers: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",  # "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

print("Without distances")

f = generate_features.featureExtractor('./split_train_clean')
train_data_list = []
n_files = len(f.files)
i = 0
for file in f.files:
    i += 1
    if i % 100 == 0:
        print("{}/{}".format(i, n_files))
    for pair in f.subImageCounts(file, False):
        l = pair[0] + [int(pair[1])]
        train_data_list.append(l)
data_train = np.array(train_data_list)

f = generate_features.featureExtractor('./split_val_clean_balanced')
val_data_list = []
n_files = len(f.files)
i = 0
for file in f.files:
    i += 1
    if i % 100 == 0:
        print("{}/{}".format(i, n_files))
    for pair in f.subImageCounts(file, False):
        l = pair[0] + [int(pair[1])]
        val_data_list.append(l)
data_val = np.array(val_data_list)

f = generate_features.featureExtractor('./split_test_clean_balanced')
test_data_list = []
n_files = len(f.files)
i = 0
for file in f.files:
    i += 1
    if i % 100 == 0:
        print("{}/{}".format(i, n_files))
    for pair in f.subImageCounts(file, False):
        l = pair[0] + [int(pair[1])]
        test_data_list.append(l)
data_test = np.array(test_data_list)

# Process data
X_train = preprocessing.scale(data_train[:, :-1])
Y_train = data_train[:, -1]
X_val = preprocessing.scale(data_val[:, :-1])
Y_val = data_val[:, -1]
X_test = preprocessing.scale(data_test[:, :-1])
Y_test = data_test[:, -1]
for name, clf in zip(names, classifiers):
    print("Fitting classifier", name, "...")
    clf.fit(X_train, Y_train)
    print("Fitted")
    print("Train Score: ", clf.score(X_train, Y_train))
    print("Val Score: ", clf.score(X_val, Y_val))
    print("Test Score: ", clf.score(X_test, Y_test))

# With distances

print("With distances")

f = generate_features.featureExtractor('./split_train_clean')
train_data_list = []
n_files = len(f.files)
i = 0
for file in f.files:
    i += 1
    if i % 100 == 0:
        print("{}/{}".format(i, n_files))
    for pair in f.subImageCounts(file, True):
        l = pair[0] + [int(pair[1])]
        train_data_list.append(l)
data_train = np.array(train_data_list)

f = generate_features.featureExtractor('./split_val_clean_balanced')
val_data_list = []
n_files = len(f.files)
i = 0
for file in f.files:
    i += 1
    if i % 100 == 0:
        print("{}/{}".format(i, n_files))
    for pair in f.subImageCounts(file, True):
        l = pair[0] + [int(pair[1])]
        val_data_list.append(l)
data_val = np.array(val_data_list)

f = generate_features.featureExtractor('./split_test_clean_balanced')
test_data_list = []
n_files = len(f.files)
i = 0
for file in f.files:
    i += 1
    if i % 100 == 0:
        print("{}/{}".format(i, n_files))
    for pair in f.subImageCounts(file, True):
        l = pair[0] + [int(pair[1])]
        test_data_list.append(l)
data_test = np.array(test_data_list)

# Process data
X_train = preprocessing.scale(data_train[:, :-1])
Y_train = data_train[:, -1]
X_val = preprocessing.scale(data_val[:, :-1])
Y_val = data_val[:, -1]
X_test = preprocessing.scale(data_test[:, :-1])
Y_test = data_test[:, -1]
for name, clf in zip(names, classifiers):
    print("Fitting classifier", name, "...")
    clf.fit(X_train, Y_train)
    print("Fitted")
    print("Train Score: ", clf.score(X_train, Y_train))
    print("Val Score: ", clf.score(X_val, Y_val))
    print("Test Score: ", clf.score(X_test, Y_test))

"""
# Macro-1
print("Macro-1")

f_train = generate_features.featureExtractor('./split_train_clean')
train_data_list = []
n_files = len(f_train.files)
i = 0
for file in f_train.files:
    i+=1
    if i%100==0:
        print("{}/{}".format(i, n_files))
    for pair in f_train.macroImageCounts(file):
        l = pair[0]+[int(pair[1])]
        train_data_list.append(l)
data_train = np.array(train_data_list)

f_val = generate_features.featureExtractor('./split_val_clean_balanced')
val_data_list = []
n_files = len(f_val.files)
i = 0
for file in f_val.files:
    i+=1
    if i%100==0:
        print("{}/{}".format(i, n_files))
    for pair in f_val.macroImageCounts(file):
        l = pair[0]+[int(pair[1])]
        val_data_list.append(l)
data_val = np.array(val_data_list)

f_test = generate_features.featureExtractor('./split_test_clean_balanced')
test_data_list = []
n_files = len(f_test.files)
i = 0
for file in f_test.files:
    i += 1
    if i % 100 == 0:
        print("{}/{}".format(i, n_files))
    for pair in f_test.macroImageCounts(file):
        l = pair[0] + [int(pair[1])]
        test_data_list.append(l)
data_test = np.array(test_data_list)

# Process data
X_train = preprocessing.scale(data_train[:, :-1])
Y_train = data_train[:, -1]
X_val = preprocessing.scale(data_val[:, :-1])
Y_val = data_val[:, -1]
X_test = preprocessing.scale(data_test[:, :-1])
Y_test = data_test[:, -1]
for name, clf in zip(names, classifiers):
    print("Fitting classifier", name, "...")
    clf.fit(X_train, Y_train)
    print("Fitted")
    print("Train Score: ", clf.score(X_train,Y_train))
    print("Val Score: ", clf.score(X_val, Y_val))
    print("Test Score: ", clf.score(X_test, Y_test))


# Macro-2
print("Macro-2")
train_data_list = []
n_files = len(f_train.files)
i = 0
for file in f_train.files:
    i+=1
    if i%100==0:
        print("{}/{}".format(i, n_files))
    for pair in f_train.macroImageCounts(file, useDists = True):
        l = pair[0]+[int(pair[1])]
        train_data_list.append(l)
data_train = np.array(train_data_list)


val_data_list = []
n_files = len(f_val.files)
i = 0
for file in f_val.files:
    i+=1
    if i%100==0:
        print("{}/{}".format(i, n_files))
    for pair in f_val.macroImageCounts(file, useDists = True):
        l = pair[0]+[int(pair[1])]
        val_data_list.append(l)
data_val = np.array(val_data_list)


test_data_list = []
n_files = len(f_test.files)
i = 0
for file in f_test.files:
    i += 1
    if i % 100 == 0:
        print("{}/{}".format(i, n_files))
    for pair in f_test.macroImageCounts(file, useDists = True):
        l = pair[0] + [int(pair[1])]
        test_data_list.append(l)
data_test = np.array(test_data_list)

# Process data
X_train = preprocessing.scale(data_train[:, :-1])
Y_train = data_train[:, -1]
X_val = preprocessing.scale(data_val[:, :-1])
Y_val = data_val[:, -1]
X_test = preprocessing.scale(data_test[:, :-1])
Y_test = data_test[:, -1]
for name, clf in zip(names, classifiers):
    print("Fitting classifier", name, "...")
    clf.fit(X_train, Y_train)
    print("Fitted")
    print("Train Score: ", clf.score(X_train,Y_train))
    print("Val Score: ", clf.score(X_val, Y_val))
    print("Test Score: ", clf.score(X_test, Y_test))


# Macro-3
print("Macro-3")

train_data_list = []
n_files = len(f_train.files)
i = 0
for file in f_train.files:
    i+=1
    if i%100==0:
        print("{}/{}".format(i, n_files))
    for pair in f_train.macroImageCounts(file, useDirs = True):
        l = pair[0]+[int(pair[1])]
        train_data_list.append(l)
data_train = np.array(train_data_list)

val_data_list = []
n_files = len(f_val.files)
i = 0
for file in f_val.files:
    i+=1
    if i%100==0:
        print("{}/{}".format(i, n_files))
    for pair in f_val.macroImageCounts(file, useDirs = True):
        l = pair[0]+[int(pair[1])]
        val_data_list.append(l)
data_val = np.array(val_data_list)

test_data_list = []
n_files = len(f_test.files)
i = 0
for file in f_test.files:
    i += 1
    if i % 100 == 0:
        print("{}/{}".format(i, n_files))
    for pair in f_test.macroImageCounts(file, useDirs = True):
        l = pair[0] + [int(pair[1])]
        test_data_list.append(l)
data_test = np.array(test_data_list)

# Process data
X_train = preprocessing.scale(data_train[:, :-1])
Y_train = data_train[:, -1]
X_val = preprocessing.scale(data_val[:, :-1])
Y_val = data_val[:, -1]
X_test = preprocessing.scale(data_test[:, :-1])
Y_test = data_test[:, -1]
for name, clf in zip(names, classifiers):
    print("Fitting classifier", name, "...")
    clf.fit(X_train, Y_train)
    print("Fitted")
    print("Train Score: ", clf.score(X_train,Y_train))
    print("Val Score: ", clf.score(X_val, Y_val))
    print("Test Score: ", clf.score(X_test, Y_test))


# Macro-4
print("Macro-4")

train_data_list = []
n_files = len(f_train.files)
i = 0
for file in f_train.files:
    i+=1
    if i%100==0:
        print("{}/{}".format(i, n_files))
    for pair in f_train.macroImageCounts(file, useDists = True,useDirs = True):
        l = pair[0]+[int(pair[1])]
        train_data_list.append(l)
data_train = np.array(train_data_list)

val_data_list = []
n_files = len(f_val.files)
i = 0
for file in f_val.files:
    i+=1
    if i%100==0:
        print("{}/{}".format(i, n_files))
    for pair in f_val.macroImageCounts(file, useDists = True, useDirs = True):
        l = pair[0]+[int(pair[1])]
        val_data_list.append(l)
data_val = np.array(val_data_list)


test_data_list = []
n_files = len(f_test.files)
i = 0
for file in f_test.files:
    i += 1
    if i % 100 == 0:
        print("{}/{}".format(i, n_files))
    for pair in f_test.macroImageCounts(file, useDists = True, useDirs = True):
        l = pair[0] + [int(pair[1])]
        test_data_list.append(l)
data_test = np.array(test_data_list)

# Process data
X_train = preprocessing.scale(data_train[:, :-1])
Y_train = data_train[:, -1]
X_val = preprocessing.scale(data_val[:, :-1])
Y_val = data_val[:, -1]
X_test = preprocessing.scale(data_test[:, :-1])
Y_test = data_test[:, -1]
for name, clf in zip(names, classifiers):
    print("Fitting classifier", name, "...")
    clf.fit(X_train, Y_train)
    print("Fitted")
    print("Train Score: ", clf.score(X_train,Y_train))
    print("Val Score: ", clf.score(X_val, Y_val))
    print("Test Score: ", clf.score(X_test, Y_test))
"""

"""
Results obtained

Without distances
Fitting classifier Nearest Neighbors ...
Fitted
Score:  0.754199906147
Fitting classifier Linear SVM ...
Fitted
Score:  0.901360863444
Fitting classifier Decision Tree ...
Fitted
Score:  0.917409666823
Fitting classifier Random Forest ...
Fitted
Score:  0.925574847489
Fitting classifier Neural Net ...
Fitted
Score:  0.92698263726
Fitting classifier AdaBoost ...
Fitted
Score:  0.924542468325
Fitting classifier Naive Bayes ...
Fitted
Score:  0.834537775692
Fitting classifier QDA ...
Fitted
Score:  0.834162365087

With distances
Fitting classifier Nearest Neighbors ...
Fitted
Score:  0.846269357109
Fitting classifier Linear SVM ...
Fitted
Score:  0.941248240263
Fitting classifier Decision Tree ...
Fitted
Score:  0.925387142187
Fitting classifier Random Forest ...
Fitted
Score:  0.940497419052
Fitting classifier Neural Net ...
Fitted
Score:  0.94669169404
Fitting classifier AdaBoost ...
Fitted
Score:  0.913561708118
Fitting classifier Naive Bayes ...
Fitted
Score:  0.883810417644
Fitting classifier QDA ...
Fitted
Score:  0.883622712342


"""











