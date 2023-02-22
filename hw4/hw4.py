import numpy as np
import matplotlib.pyplot as plt
from numba import jit


def generator_of_norm(m, s):
    sum = 0
    for i in range(12):
        sum += np.random.uniform()
    return m + (s ** 0.5) * (sum - 6)


def Logistic_regression(N, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2):
    D1 = [(generator_of_norm(mx1, vx1), generator_of_norm(my1, vy1)) for _ in range(N)]
    D2 = [(generator_of_norm(mx2, vx2), generator_of_norm(my2, vy2)) for _ in range(N)]

    x = np.zeros((2 * N, 3))
    y = np.zeros((2 * N, 1))
    for i in range(2 * N):
        x[i][0] = 1
        if i < N:
            x[i][1] = D1[i][0]
            x[i][2] = D1[i][1]
        else:
            x[i][1] = D2[i - N][0]
            x[i][2] = D2[i - N][1]
            y[i][0] = 1

    def compute_metrics(x, y, w):
        confusion_matrix = np.zeros((2, 2))

        for x, y in zip(x, y):
            if 1 / (1 + np.exp(-x.dot(w))) > 0.5:
                if y == 1:
                    confusion_matrix[1][1] += 1
                else:
                    confusion_matrix[0][1] += 1
            else:
                if y == 1:
                    confusion_matrix[1][0] += 1
                else:
                    confusion_matrix[0][0] += 1

        sensitivity = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
        specificity = confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])

        for i in range(3):
            print(f'{w[i][0]:15.10f}')

        print('\nConfusion Matrix:')
        print('             Predict cluster 1 Predict cluster 2')
        print('Is cluster 1        %d                %d' % (confusion_matrix[0][0], confusion_matrix[0][1]))
        print('Is cluster 2        %d                %d' % (confusion_matrix[1][0], confusion_matrix[1][1]))
        print('\nSensitivity (Successfully predict cluster 1): %.5f' % sensitivity)
        print('Specificity (Successfully predict cluster 2): %.5f' % specificity)


    def gradient_descent(x, y, lr=0.001, e=10e-4):
        w_last, w = np.zeros((3, 1)), np.zeros((3, 1))
        while True:
            w = w_last + lr * x.T.dot((y - 1 / (1 + np.exp(-x.dot(w_last)))))
            if sum(abs(w - w_last)) < e:
                break
            w_last = w
        
        compute_metrics(x, y, w)
        return w
    
    def Newton_method(x, y, N, lr=0.001, e=10e-4):
        w_last, w = np.zeros((3, 1)), np.zeros((3, 1))

        while True:
            gradient = x.T.dot((y - 1 / (1 + np.exp(-x.dot(w_last)))))
            D = np.zeros((2 * N, 2 * N))
            for i in range(2 * N):
                D[i][i] = np.exp(-x[i].dot(w_last)) / ((1 + np.exp(-x[i].dot(w_last))) ** 2)
            H = x.T.dot(D).dot(x)

            if np.linalg.det(H) == 0:
                w = w_last + lr * gradient
            else:
                w = w_last + lr * np.linalg.inv(H).dot(gradient)

            if sum(abs(w - w_last)) < e:
                break
            w_last = w

        compute_metrics(x, y, w)
        return w
    
    print('Gradient descent:\n')
    w_gradient = gradient_descent(x, y)
    print('\n--------------------------------------------')
    print("Newton's method:\n")
    w_newton = Newton_method(x, y, N)

    plt.subplot(131)
    plt.title('Ground truth')
    plt.scatter([d[0] for d in D1], [d[1] for d in D1], c='r')
    plt.scatter([d[0] for d in D2], [d[1] for d in D2], c='b')

    plt.subplot(132)
    plt.title('Gradient descent')
    predict_0, predict_1 = [], []
    for x_v in x:
        if 1 / (1 + np.exp(-x_v.dot(w_gradient))) > 0.5:
            predict_1.append((x_v[1], x_v[2]))
        else:
            predict_0.append((x_v[1], x_v[2]))
    plt.scatter([i[0] for i in predict_0], [i[1] for i in predict_0], c='r')
    plt.scatter([i[0] for i in predict_1], [i[1] for i in predict_1], c='b')
    
    plt.subplot(133)
    plt.title("Newton's method")
    predict_0, predict_1 = [], []
    for x_v in x:
        if 1 / (1 + np.exp(-x_v.dot(w_newton))) > 0.5:
            predict_1.append((x_v[1], x_v[2]))
        else:
            predict_0.append((x_v[1], x_v[2]))
    plt.scatter([i[0] for i in predict_0], [i[1] for i in predict_0], c='r')
    plt.scatter([i[0] for i in predict_1], [i[1] for i in predict_1], c='b')
    plt.show()


def EM(training_data_path, training_label_path):
    with open(training_data_path, mode = 'rb') as f:
        fileContent = f.read()

    number_of_training_images = int.from_bytes(fileContent[4:8], 'big')
    n = int.from_bytes(fileContent[8:12], 'big')
    m = int.from_bytes(fileContent[12:16], 'big')

    training_images = []
    for i in range(number_of_training_images):
        training_images.append(fileContent[i * n * m + 16:(i + 1) * n * m + 16])

    temp = training_images.copy()
    training_images = np.zeros((number_of_training_images, n * m))
    for i in range(number_of_training_images):
        for j in range(n * m):
            training_images[i][j] = temp[i][j]
    
        
    with open(training_label_path, mode = 'rb') as f:
        fileContent = f.read()
        
    training_labels = []
    for i in range(number_of_training_images):
        training_labels.append(fileContent[8 + i])

    temp = training_labels.copy()
    training_labels = np.zeros((number_of_training_images), dtype=np.int32)
    for i in range(number_of_training_images):
        training_labels[i] = temp[i]

    lamda = np.full((10), 0.1)
    prob = np.random.rand(10, n * m)
    
    @jit
    def E(dataset_size, n, m, lamda, prob, training_images):
        w = np.zeros((dataset_size, 10))
        for i in range(dataset_size):
            for j in range(10):
                w[i][j] = lamda[j]
                for k in range(n * m):
                    w[i][j] *= (prob[j][k] ** (training_images[i][k] >= 128))
                    w[i][j] *= ((1 - prob[j][k]) ** (1 - (training_images[i][k] >= 128)))
            if sum(w[i]) != 0:
                    w[i] = w[i] / sum(w[i])
        return w
    
    @jit
    def M(dataset_size, n, m, w, training_images):
        lamda, prob = np.zeros((10)), np.zeros((10, n * m))
        w_sum = np.zeros((10, n * m))
        for i in range(dataset_size):
            for j in range(10):
                lamda[j] += w[i][j]
                for k in range(n * m):
                    prob[j][k] += (w[i][j] * (training_images[i][k] >= 128))
                    w_sum[j][k] += w[i][j]

        lamda /= dataset_size

        for i in range(10):
            for j in range(n * m):
                if w_sum[i][j] != 0:
                    prob[i][j] /= w_sum[i][j]
        
        return lamda, prob

    prob_last = prob
    count = 0
    while True:
        count += 1
        w = E(number_of_training_images, n, m, lamda, prob_last, training_images)
        lamda, prob = M(number_of_training_images, n, m, w, training_images)

        for j in range(10):
            print(f'class {j}:')
            for k in range(n * m):
                if prob[j][k] > 0.5:
                    print('1 ', end='')
                else:
                    print('0 ', end='')
                if k % 28 == 27:
                    print()
            print()
        
        diff = sum(sum(abs(prob - prob_last)))
        print(f'No. of Iteration: {count}, Difference: {diff}')
        prob_last = prob

        if diff < 5:
            break

    print('------------------------------------------------------------.')
    print('------------------------------------------------------------.\n')
    
    @jit
    def match_label(dataset_size, training_images, training_labels, n, m, lamda, prob):
        count, matching = np.zeros((10, 10)), np.zeros((10), dtype=np.int32)
        for i in range(dataset_size):
            p = np.zeros(10)
            for j in range(10):
                p[j] = lamda[j]
                for k in range(n * m):
                    p[j] *= (prob[j][k] ** (training_images[i][k] >= 128))
                    p[j] *= ((1 - prob[j][k]) ** (1 - (training_images[i][k] >= 128)))
            count[training_labels[i]][np.argmax(p)] += 1
        for _ in range(10):
            idx = np.argmax(count)
            label_class = idx // 10
            predict_class = idx % 10
            matching[label_class] = predict_class
            count[label_class, :] = -1
            count[:, predict_class] = -1
        return matching


    matching = match_label(number_of_training_images, training_images, training_labels, n, m, lamda, prob)
    
    for i in range(10):
        label = matching[i]
        print(f'labeled class {i}:')
        for j in range(n * m):
            if prob[label][j] > 0.5:
                print('1 ', end='')
            else:
                print('0 ', end='')
            if j % 28 == 27:
                print()
        print()


    @jit
    def compute_metrics(dataset_size, training_images, training_labels, n, m, lamda, prob):
        correct = 0
        confusion_matrix = np.zeros((10, 2, 2))
        for i in range(dataset_size):
            p = np.zeros(10)
            for j in range(10):
                p[j] = lamda[j]
                for k in range(n * m):
                    p[j] *= (prob[j][k] ** (training_images[i][k] >= 128))
                    p[j] *= ((1 - prob[j][k]) ** (1 - (training_images[i][k] >= 128)))
            predict_class = np.argmax(p)

            training_labels[i] = matching[training_labels[i]]
            for j in range(10):
                label = matching[j]
                if predict_class == label:
                    if label == training_labels[i]:
                        confusion_matrix[j][0][0] += 1
                    else:
                        confusion_matrix[j][1][0] += 1
                else:
                    if label == training_labels[i]:
                        confusion_matrix[j][0][1] += 1
                    else:
                        confusion_matrix[j][1][1] += 1

        return confusion_matrix

    confusion_matrix = compute_metrics(number_of_training_images, training_images, training_labels, n, m, lamda, prob)

    correct = 0
    for i in range(10):
        print('------------------------------------------------------------.\n')

        sensitivity = confusion_matrix[i][0][0] / (confusion_matrix[i][0][0] + confusion_matrix[i][0][1])
        specificity = confusion_matrix[i][1][1] / (confusion_matrix[i][1][0] + confusion_matrix[i][1][1])

        print('\nConfusion Matrix %d:' % i)
        print('                 Predict number %d Predict not number %d' % (i, i))
        print('Is number %d           %d                  %d' % (i, confusion_matrix[i][0][0], confusion_matrix[i][0][1]))
        print("Isn't number %d        %d                  %d" % (i, confusion_matrix[i][1][0], confusion_matrix[i][1][1]))
        print(f'\nSensitivity (Successfully predict numbeer {i}): %.5f' % sensitivity)
        print(f'Specificity (Successfully not predict number {i}): %.5f\n' % specificity)

        correct += confusion_matrix[i][0][0]

    print(f'Total iteration to converge: {count}')
    print(f'Total error rate: {1 - correct / number_of_training_images}')
   

if __name__ == '__main__':
    '''
    parameter
    '''
    N = 50
    mx1, my1 = 1, 1
    mx2, my2 = 3, 3
    vx1, vy1 = 2, 2
    vx2, vy2 = 4, 4
    Logistic_regression(N, mx1, my1, mx2, my2, vx1, vy1, vx2, vy2)

    '''
    parameter
    '''
    training_data_path = './train-images.idx3-ubyte'
    training_label_path = './train-labels.idx1-ubyte'
    # EM(training_data_path, training_label_path)
