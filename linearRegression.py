import numpy as np
import matplotlib.pyplot as plt

# linear regression
X = np.loadtxt('task_1_capital.txt', delimiter='\t\t', skiprows=1, dtype=int)
Y = np.copy(X[:, 1])
X[:, 1] = 1.0
res = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(Y)

for capital, rental in zip(X[:, 0], Y):
    plt.scatter(capital, rental, c='r', marker='.')

# drawing graphic
x = [60000., 325000.]
y = [res[0] * x[0] + res[1], res[0] * x[1] + res[1]]

plt.plot(x, y)

plt.xlabel('Capital')
plt.ylabel('Rental')
plt.show()
