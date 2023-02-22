import time
import matplotlib.pyplot as plt
import numpy as np
from libsvm.svmutil import *

X_train = np.zeros((5000, 784), dtype=float)
Y_train = np.zeros((5000), dtype=int)
X_test = np.zeros((2500, 784), dtype=float)
Y_test = np.zeros((2500), dtype=int)

#############################################################################
# Load Data
#############################################################################
with open('./data/X_train.csv') as f:
    for i, line in enumerate(f):
        for j, pixel in enumerate(line.replace('\n', '').split(',')):
            X_train[i, j] = float(pixel)

with open('./data/Y_train.csv') as f:
    for i, line in enumerate(f):
        target = line.replace('\n', '')
        Y_train[i] = int(target)

with open('./data/X_test.csv') as f:
    for i, line in enumerate(f):
        for j, pixel in enumerate(line.replace('\n', '').split(',')):
            X_test[i, j] = float(pixel)

with open('./data/Y_test.csv') as f:
    for i, line in enumerate(f):
        target = line.replace('\n', '')
        Y_test[i] = int(target)


#############################################################################
# Define function
#############################################################################

def LinearKernel(X):
    return X.dot(X.T)


def RBFKernel(X, gamma):
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist = X[i] - X[j]
            dist = dist.T.dot(dist)
            K[i, j] = np.exp(-gamma * dist)
    return K


def svm(X_train, Y_train, X_test, Y_test, parameter):
    m =  svm_train(Y_train, X_train, parameter)
    p_labs, p_acc, p_vals = svm_predict(Y_test, X_test, m)
    return p_acc


def compare_para (X, Y, para, best_acc, best_para):
    acc = svm_train(Y, X, para)
    if acc > best_acc:
        return acc, para
    return best_acc, best_para


def gridSearch(X, Y, kernel):
    best_acc, best_para = 0, ""
    C = [0.001, 0.01, 0.1, 1, 10]
    gammas = [1/784, 0.01, 0.1, 1]
    degrees = [2, 3, 4]
    if kernel == 0:
        for c in C:
            para = f'-t {kernel} -c {c} -q -v 5'
            print(para)
            best_acc, best_para = compare_para(X, Y, para, best_acc, best_para)
    elif kernel == 1:
        for c in C:
            for gamma in gammas:
                for degree in degrees:
                    para = f'-t {kernel} -c {c} -g {gamma} -d {degree} -q -v 5'
                    print(para)
                    best_acc, best_para = compare_para(X, Y, para, best_acc, best_para)
    elif kernel == 2:
        for c in C:
            for gamma in gammas:
                para = f'-t {kernel} -c {c} -g {gamma} -q -v 5'
                print(para)
                best_acc, best_para = compare_para(X, Y, para, best_acc, best_para)
    print('\n-------------------------------------------------------------')
    print(f'Best cross validation accuracy: {best_acc}')
    print(f'Best parameter: {best_para}')
    print('-------------------------------------------------------------\n')
    return best_acc, best_para


##################################################################
# Part 1
##################################################################
print('Part1:\n')

print('linear:')
linear_acc = svm(X_train, Y_train, X_test, Y_test, '-t 0 -q')

print('\npolynomial:')
polynomial_acc = svm(X_train, Y_train, X_test, Y_test, '-t 1 -d 2 -q')

print('\nrbf:')
rbf_acc = svm(X_train, Y_train, X_test, Y_test, '-t 2 -q')

print()

##################################################################
# Part 2
##################################################################
print('Part2:\n')

print('linear:')
l_acc, l_para = gridSearch(X_train, Y_train, 0)
# 去掉 -v parameter
best_para = l_para[:-5]
svm(X_train, Y_train, X_test, Y_test, best_para)

print('\npolynomial:')
p_acc, p_para = gridSearch(X_train, Y_train, 1)
# 去掉 -v parameter
best_para = p_para[:-5]
svm(X_train, Y_train, X_test, Y_test, best_para)

print('\nradial basis function:')
r_acc, r_para = gridSearch(X_train, Y_train, 2)
# 去掉 -v parameter
best_para = r_para[:-5]
svm(X_train, Y_train, X_test, Y_test, best_para)

print()

##################################################################
# Part 3
##################################################################
gamma = 1 / 784
train_kernel = LinearKernel(X_train) + RBFKernel(X_train, gamma)
test_kernel = LinearKernel(X_test) + RBFKernel(X_test, gamma)
train_kernel = np.hstack((np.arange(1, len(Y_train)+1).reshape(-1, 1), train_kernel))
test_kernel = np.hstack((np.arange(1, len(Y_test)+1).reshape(-1, 1), test_kernel))
m = svm_train(Y_train, train_kernel, '-t 4 -q')
labs, acc, vals = svm_predict(Y_test, test_kernel, m)