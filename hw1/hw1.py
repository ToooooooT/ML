import numpy as np
import matplotlib.pyplot as plt

class LSE:
    def __init__(self, path, n, Lambda):
        self.path = path
        self.n = n
        self.Lambda = Lambda
        self.x = []
        self.y = []

    def getData (self):
        with open (self.path, 'r') as f:
            for line in f.readlines():
                data = line.replace('\n', '').split(',')
                self.x.append(float(data[0]))
                self.y.append(float(data[1]))

        self.A = np.zeros(shape = (len(self.x), self.n))
        self.b = np.array(self.y).reshape((len(self.y), 1))
        
        for idx, x in enumerate(self.x):
            for exp in range(self.n):
                self.A[idx, exp] = x ** exp

def f (x, weights):
    n = len(weights)
    y = [0] * len(x)
    for idx, num in enumerate(x):
        for i in range(n):
            y[idx] += ((num ** i) * weights[i, 0])
    return y
        
solve = LSE('./testfile.txt', 3, 10000)
solve.getData()

inv = np.linalg.inv(solve.A.T.dot(solve.A) + solve.Lambda * np.identity(solve.n))
weights = inv.dot(solve.A.T).dot(solve.b)

print('Fitting line: ', end = '')
for i in range(solve.n - 1, 0, -1):
    print(f'{weights[i, 0]:.13f}X^{i} + ', end = '')
print(f'{weights[0, 0]:.13f}')

diff = solve.A.dot(weights) - solve.b
print(f'Total error: {diff.T.dot(diff)[0, 0]:.6f}')

plt.scatter(solve.x, solve.y, color = 'red')
x = np.arange(min(solve.x), max(solve.x), 0.01)
y = f(x, weights)
plt.plot(x, y, color = 'black')
plt.show()