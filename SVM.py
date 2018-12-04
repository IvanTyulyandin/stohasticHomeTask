from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


data = np.loadtxt("chips.txt", delimiter=',')

parametersSVC = {'kernel': ('linear', 'rbf', 'poly'),
                 'C': np.arange(1., 10., 0.1),
                 'gamma': ('scale', 'auto')}

parametersKNN = {'n_neighbors': range(1, 10),
                 'weights': ('uniform', 'distance'),
                 'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute')}

metrics = ['average_precision', 'balanced_accuracy']
titles = ['SVC_av_prec', 'kNN_av_prec', 'SVC_bal_accur', 'kNN_bal_accur']
models = []
params = [parametersSVC, parametersKNN]
algos = [svm.SVC(), KNeighborsClassifier()]

for metric in metrics:
    for param, algo in zip(params, algos):
        clf = GridSearchCV(algo, param, cv=5, scoring=metric)
        clf.fit(data[:, :2], data[:, 2])
        models.append(clf.best_estimator_)


# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = data[:, 0], data[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=data[:, 2], cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
