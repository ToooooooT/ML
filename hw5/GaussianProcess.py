import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

X = []
Y = []
with open('./data/input.data') as f:
    for line in f:
        x, y = line.replace('\n', '').split(' ')
        X.append(float(x))
        Y.append(float(y))

X = np.array(X).reshape(-1, 1)
Y = np.array(Y).reshape(-1, 1)

def RationalQuadraticKernel(x1, x2, sigma, alpha, l):
    '''
    sigma ^ 2 : the overall variance (sigma is also known as amplitude)
    l : the lengthscale
    alpha : the scale-mixture (alpha > 0)
    '''
    return (sigma ** 2) * ((1 + ((x1 - x2) ** 2) / (2 * alpha * (l ** 2))) ** (-alpha))


def covariance(X, beta, sigma, alpha, l):
    n = X.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = RationalQuadraticKernel(X[i], X[j], sigma, alpha, l) + (beta ** (-1)) * (i == j)
    return C

def GaussianProcessRegression(n, beta, X, Y, sigma, alpha, l):
    '''
    n : number of test data points
    beta : var^-1 of error gaussian distribution
    '''
    mean = np.zeros((n))
    var = np.zeros((n))
    x = np.linspace(-60, 60, n)
    C = covariance(X, beta, sigma, alpha, l)

    for i in range(n):
        k = np.zeros((34, 1))
        for j in range(34):
            k[j, 0] = RationalQuadraticKernel(x[i], X[j, 0], sigma, alpha, l)
        mean[i] = k.T.dot(np.linalg.inv(C)).dot(Y)
        var[i] = RationalQuadraticKernel(x[i], x[i], sigma, alpha, l) + (beta ** (-1)) - k.T.dot(np.linalg.inv(C)).dot(k)

    return x, mean, var

    
def NegativeMarginalLikelihood(theta, X, Y, beta):
    C = covariance(X, beta, theta[0], theta[1], theta[2])
    likelihood = 0.5 * np.log(np.linalg.det(C)) + 0.5 * Y.T.dot(np.linalg.inv(C)).dot(Y)
    return likelihood[0]

sigma, alpha, l = 1, 1, 1
beta = 5

x, mean, var = GaussianProcessRegression(1000, beta, X, Y, sigma, alpha, l)

# Optimize parameters
opt = minimize(NegativeMarginalLikelihood, np.array([sigma, alpha, l]), args=(X, Y, beta))

sigma = opt.x[0]
alpha = opt.x[1]
l = opt.x[2]

print(f'optimize sigma: {sigma:.4f}')
print(f'optimize alpha: {alpha:.4f}')
print(f'optimize l: {l:.4f}')

opt_x, opt_mean, opt_var = GaussianProcessRegression(1000, beta, X, Y, sigma, alpha, l)

fig = plt.figure()
interval = 1.96 * (var ** 0.5)

ax = fig.add_subplot(1, 2, 1)
ax.set_title('without optimize kernel parameters')
ax.plot(X, Y, "k.")
ax.plot(x, mean, "red")
ax.fill_between(x, mean + interval, mean - interval, color='lightgreen')
ax.set_xlim([-60, 60])
ax.set_ylim([-5, 5])

opt_interval = 1.96 * (opt_var ** 0.5)

ax = fig.add_subplot(1, 2, 2)
ax.set_title('optimize kernel parameters')
ax.plot(X, Y, "k.")
ax.plot(opt_x, opt_mean, "red")
ax.fill_between(x, opt_mean + opt_interval, opt_mean - opt_interval, color='lightgreen')
ax.set_xlim([-60, 60])
ax.set_ylim([-5, 5])

plt.show()







