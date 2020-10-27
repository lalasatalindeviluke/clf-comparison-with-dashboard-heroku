from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.linear_model import RidgeClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, \
ExtraTreesClassifier, RandomForestClassifier
from xgboost import XGBClassifier


mnist = fetch_openml('mnist_784', version=1, cache=True)
X = mnist["data"]
y = mnist["target"]

X_train = X[:60000]
y_train = y[:60000]
X_test = X[60000:]
y_test = y[60000:]

sgd_clf = SGDClassifier(random_state=42)
softmax_clf = LogisticRegression(multi_class="multinomial", random_state=42)
ridge_clf = RidgeClassifier(random_state=42)
perceptron_clf = Perceptron(random_state=42)
kn_clf = KNeighborsClassifier(n_neighbors=10)
linsvc_clf = LinearSVC(random_state=42)
svc_clf = SVC(random_state=42)
decisiontree_clf = DecisionTreeClassifier(random_state=42)
#Ensemble
adaboost_clf = AdaBoostClassifier(random_state=42)
gb_clf = GradientBoostingClassifier(random_state=42)
extratree_clf = ExtraTreesClassifier(random_state=42)
randforest_clf = RandomForestClassifier(random_state=42)
xgb_clf = XGBClassifier(random_state=42)

clf_list = [sgd_clf, log_clf, ridge_clf, perceptron_clf,
            kn_clf, linsvc_clf, svc_clf, decisiontree_clf,
            adab_clf, gb_clf, extratree_clf, randforest_clf,
            xgb_clf]


sgd_clf.fit(X_train, y_train)
y_pred = sgd_clf.predict(X_test)
accuracy_score(y_test, y_pred) #0.874

softmax_clf.fit(X_train, y_train)
y_pred = softmax_clf.predict(X_test)
accuracy_score(y_test, y_pred) #0.9255

ridge_clf.fit(X_train, y_train)
y_pred = ridge_clf.predict(X_test)
accuracy_score(y_test, y_pred) #0.8603

perceptron_clf.fit(X_test, y_test)
y_pred = perceptron_clf.predict(X_test)
accuracy_score(y_test, y_pred) #0.8953

kn_clf.fit(X_train, y_train)
y_pred = kn_clf.predict(X_test)
accuracy_score(y_test, y_pred) #0.9665

linsvc_clf.fit(X_train, y_train)
y_pred = linsvc_clf.predict(X_test)
accuracy_score(y_test, y_pred)  #0.8236

svc_clf.fit(X_train, y_train)
y_pred = svc_clf.predict(X_test)
accuracy_score(y_test, y_pred) #0.9792

decisiontree_clf.fit(X_train, y_train)
y_pred = decisiontree_clf.predict(X_test)
accuracy_score(y_test, y_pred) #0.8755

adaboost_clf.fit(X_train, y_train)
y_pred = adaboost_clf.predict(X_test)
accuracy_score(y_test, y_pred) #0.7299

# gb_clf.fit(X_train, y_train)
# y_pred = gb_clf.predict(X_test)
# accuracy_score(y_test, y_pred)

extratree_clf.fit(X_train, y_train)
y_pred = extratree_clf.predict(X_test)
accuracy_score(y_test, y_pred) #0.9722

randforest_clf.fit(X_train, y_train)
y_pred = randforest_clf.predict(X_test)
accuracy_score(y_test, y_pred) #0.9705

xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)
accuracy_score(y_test, y_pred) #0.978

import pandas as pd
clf_test_df = pd.DataFrame({'Classifier': ['SGDClassifier', 'LogisticRegression',
                                      'RidgeClassifier', 'Perceptron',
                                      'KNeighborsClassifier', 'LinearSVC',
                                      'SVC', 'DecisionTreeClassifier',
                                      'AdaBoostClassifier', 'ExtraTreesClassifier',
                                      'RandomForestClassifier', 'XGBClassifier'],
                       'Accuracy Score': [0.874, 0.9255, 0.8603, 0.8953,
                                          0.9665, 0.8236, 0.9792, 0.8755,
                                          0.7299, 0.9722, 0.9705, 0.978],
                       'Ensemble': [False, False, False, False,
                                    False, False, False, False,
                                    True, True, True, True]
                       })
clf_test_df = clf_test_df.sort_values(by=['Accuracy Score'])
clf_test_df["index"] = [i for i in range(12)]
clf_test_df = clf_test_df.set_index("index")

clf_test_df.to_csv("D:\\研究方法期中報告\\clf-test.csv", index=False)

import plotly.express as px
from plotly.offline import plot
fig = px.bar(clf_test_df, x='Classifier', y='Accuracy Score', range_y=[0,1],
             color='Ensemble', title="Comparison of Classifier")
fig.write_html("E:\\spyder-py3\\機器學習\\bar.html")
plot(fig)

