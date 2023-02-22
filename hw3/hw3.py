import numpy as np
import matplotlib.pyplot as plt

def generator_of_norm(m, s):
    sum = 0
    for i in range(12):
        sum += np.random.uniform()
    return m + (s ** 0.5) * (sum - 6)


def poly_basis_data_generator(n, a, w):
    rand = np.random.uniform(-1, 1)
    x = np.zeros((1, n))
    for i in range(n):
        x[0][i] = rand ** i
    e = generator_of_norm(0, a)
    y = x.dot(w) + e
    return x, y



def Sequential_Estimator():
    '''
    parameter
    '''
    m = 3
    s = 5

    mean, var, n = 0.0, 0.0, 0

    def add_point(mean, var, n, val):
        new_mean = val / (n + 1) + n / (n + 1) * mean 
        if n == 0:
            var = 0
        else:
            var = 1 / (n + 1) * ((val - new_mean) ** 2 + n * var + n * ((mean - new_mean) ** 2))
        return new_mean, var, n + 1

    print(f'Data point source function: N({m}, {s})')

    val = generator_of_norm(m, s)
    while(1):
        print(f'Add data point: {val}')
        mean, var, n = add_point(mean, var, n, val)
        print(f'Mean = {mean}   Variance = {var}')
        if abs(mean - m) < 0.01 and abs(var - s) < 0.01:
            break
        val = generator_of_norm(m, s)


def Baysian_Linear_regression():
    '''
    parameter
    '''
    b = 1 # 1 / var
    n = 3
    a = 3 # 1 / var
    w = np.array([1, 2, 3]).reshape((n, 1))

    S = np.identity(n) * b
    M = np.zeros((n, 1))
    ten_S = np.identity(n) * b
    ten_M = np.zeros((n, 1))
    fifty_S = np.identity(n) * b
    fifty_M = np.zeros((n, 1))

    def add_data(M, S, a, x, y):
        Gamma = a * x.T.dot(x) + S
        M = np.linalg.inv(Gamma).dot(a * x.T * y + S.dot(M))
        return M, Gamma

    
    def f(x, n, M, S, a):
        y, y_top, y_low = [], [], []
        var = np.linalg.inv(S)
        for val in x:
            t = np.zeros((1, n))
            for i in range(n):
                t[0][i] = val ** i
            y.append(t.dot(M)[0][0])
            y_top.append(t.dot(M)[0][0] + 1 / a + t.dot(var).dot(t.T)[0][0])
            y_low.append(t.dot(M)[0][0] - 1 / a - t.dot(var).dot(t.T)[0][0])
        return y, y_top, y_low


    def draw(x, y, y_top, y_low, title):
        plt.title(title)
        plt.xlim(-2, 2)
        plt.ylim(-25, 25)
        plt.plot(x, y, 'black')
        plt.plot(x, y_top, 'r')
        plt.plot(x, y_low, 'r')


    def plot_predict(data, n, M, S, a, w, ten_M, ten_S, fifty_M, fifty_S):
        # data = (x, y); x = [1, x, x^2, ...]
        x = np.arange(-2, 2, 0.001)

        # Ground truth
        plt.subplot(221)
        y, y_top, y_low = [], [], []
        for val in x:
            t = np.zeros((1, n))
            for i in range(n):
                t[0][i] = val ** i
            y.append(t.dot(w)[0][0])
            y_top.append(t.dot(w)[0][0] + 1 / a)
            y_low.append(t.dot(w)[0][0] - 1 / a)
        draw(x, y, y_top, y_low, 'Ground truth')
        
        
        # predict result
        plt.subplot(222)
        y, y_top, y_low = f(x, n, M, S, a)
        draw(x, y, y_top, y_low, 'Predict result')
        plt.scatter([d[0][0][1] for d in data], [d[1] for d in data], c = "blue")

        # After 10 incomes
        plt.subplot(223)
        y, y_top, y_low = f(x, n, ten_M, ten_S, a)
        draw(x, y, y_top, y_low, 'After 10 incomes')
        plt.scatter([d[0][0][1] for d in data[:10]], [d[1] for d in data[:10]], c = "blue")

        # After 50 incomes
        plt.subplot(224)
        y, y_top, y_low = f(x, n, fifty_M, fifty_S, a)
        draw(x, y, y_top, y_low, 'After 50 incomes')
        plt.scatter([d[0][0][1] for d in data[:50]], [d[1] for d in data[:50]], c = "blue")
        plt.show()

            
    data = []

    # assume sample 10000 points to converge
    for count in range(10000):
        x, y = poly_basis_data_generator(n, a, w)

        data.append((x, y))
        print(f'Add data point ({x[0][1]}, {y})')

        M, S = add_data(M, S, a, x, y)

        print('\nPosterior mean:')
        for i in range(n):
            print(f'{M[i][0]:15.10f}')
        print('\nPosterior variance:')
        var = np.linalg.inv(S)
        for i in range(n):
            for j in range(n - 1):
                print(f'{var[i][j]:15.10f}', end = ',    ')
            print(f'{var[i][n - 1]:15.10f}')
        print()

        print(f'\nPredictive distribution ~ N({x.dot(M)[0][0]}, {1 / a + x.dot(var).dot(x.T)[0][0]})\n')

        if count == 9:
            ten_S = S
            ten_M = M       
        elif count == 49:
            fifty_S = S
            fifty_M = M

    plot_predict(data, n, M, S, a, w, ten_M, ten_S, fifty_M, fifty_S)

if __name__ == '__main__':
    # Sequential_Estimator()
    Baysian_Linear_regression()