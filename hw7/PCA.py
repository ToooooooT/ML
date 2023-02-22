import numpy as np
from scipy.spatial.distance import squareform, pdist
from array2gif import write_gif
import cv2
import os
import matplotlib.pyplot as plt

def read_img(path, size=50):
    '''
    path : images directory path
    size : resize image to size x size x 3
    return : 
        images : (# of datas, size x size x 3) ndarray
        labels : (# of datas) ndarray
    '''
    images = np.zeros((1, size ** 2))
    images_path = os.listdir(path)
    labels = np.zeros(len(images_path)).astype('uint8')
    for i, image_path in enumerate(images_path):
        labels[i] = int(image_path.split('.')[0][7:9]) - 1
        image_path = os.path.join(path, image_path)
        img = cv2.imread(image_path)
        img = cv2.resize(img, (size, size))[:, :, 0]
        img = img.reshape(1, -1)
        images = np.vstack((images, img))
    return images[1:, :], labels


def PCA (X, dim=25):
    '''
    X : ndarray (N, D) , N (number of data), D (dimension of image)
    dim : first dim largest eigenvalues 
    return :
        eigenvalues : ndarray (dim)
        eigenvectors : ndarray (D, dim)
        X_mean : ndarray (1, D)
    '''
    (n, d) = X.shape
    X_mean = np.mean(X, axis=0).reshape(1, -1)
    X = X - X_mean
    S = X.T.dot(X)
    (eigenvalues, eigenvectors) = np.linalg.eig(S)
    sort_index = np.argsort(-eigenvalues)
    sort_index = sort_index[:dim]
    eigenvalues = np.real(eigenvalues[sort_index])
    eigenvectors = np.real(eigenvectors[:, sort_index])
    eigenvectors_norm = np.linalg.norm(eigenvectors, axis=0)
    eigenvectors = eigenvectors / eigenvectors_norm

    return eigenvalues, eigenvectors, X_mean


def kernel_PCA (X, dim=25):
    '''
    X : ndarray (N, D) , N (number of data), D (dimension of image)
    dim : first dim largest eigenvalues 
    return :
        eigenvalues : ndarray (dim)
        eigenvectors : ndarray (N, dim)
        K_center : Gram matrix
    '''
    (n, d) = X.shape
    # kernel 1 (RBF)
    gamma = 0.00000057
    K1 = squareform(np.exp(-gamma * pdist(X, 'sqeuclidean')))
    # kernel 2 (polynomial)
    K2 = np.zeros((n, n))
    alpha, c, poly = 1, 1, 3

    for i in range(n):
        for j in range(n):
            K2[i, j] = (alpha * X[i, :].reshape(1, -1).dot(X[j, :].reshape(-1, 1)) + c) ** poly

    K = K2
    ones_N = np.ones((n, n)) / n
    K_mean = ones_N.dot(K) + K.dot(ones_N) - ones_N.dot(K).dot(ones_N)
    K_center = K - K_mean

    (eigenvalues, eigenvectors) = np.linalg.eig(K_center)
    sort_index = np.argsort(-eigenvalues)
    sort_index = sort_index[:dim]
    eigenvalues = np.real(eigenvalues[sort_index])
    eigenvectors = np.real(eigenvectors[:, sort_index])
    eigenvectors_norm = np.linalg.norm(eigenvectors, axis=0)
    eigenvectors = eigenvectors / eigenvectors_norm

    return eigenvalues, eigenvectors, K_center


def project_kernel_PCA (eigenvectors, train_X, test_X):
    '''
    eigenvectors : ndarray (N, dim)
    train_X : ndarray (N, D) , N (number of data), D (dimension of image)
    test_X : ndarray (N, D) , N (number of data), D (dimension of image)
    return -> ndarray (N, dim), N (number of testing data)
    '''
    X = np.vstack((train_X, test_X))
    (n, d) = X.shape
    # kernel 1 (RBF)
    gamma = 0.00000057
    K1 = squareform(np.exp(-gamma * pdist(X, 'sqeuclidean')))
    # kernel 2 (polynomial)
    K2 = np.zeros((n, n))
    alpha, c, poly = 1, 1, 3
    for i in range(n):
        for j in range(n):
            K2[i, j] = (alpha * X[i, :].reshape(1, -1).dot(X[j, :].reshape(-1, 1)) + c) ** poly

    N = train_X.shape[0]
    K_center = K2[N:, :N].copy()
    tmp = K2[N:, :N].copy()
    tmp2 = K2[:N, :N].copy()
    third_term = np.sum(np.sum(K2[:N, :N])) / N / N
    for i in range(K_center.shape[0]):
        for j in range(K_center.shape[1]):
            K_center[i, j] -= (np.sum(tmp[i, :]) / N + np.sum(tmp2[j, :]) / N - third_term)

    return K_center.dot(eigenvectors)


def show_eigenfaces(eigenvectors, k=25, size=25):
    eigenfaces = eigenvectors.T.reshape((k, size, size))

    fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(8, 10))
    for i in range(k):
        axes[i // 5][i % 5].set_title(f'{i + 1}')
        axes[i // 5][i % 5].imshow(eigenfaces[i], cmap="gray")
    plt.savefig('./first_25_eigenfaces.png')
    plt.show()


def show_reconstruct(train_X, X_reconstruct, size, num=10):
    randint = np.random.choice(train_X.shape[0], num)
    fig, axes = plt.subplots(num, 2, sharex=True, sharey=True, figsize=(8, 10))
    axes[0][0].set_title(f'original')
    axes[0][1].set_title(f'reconstruct')
    for i in range(num):
        idx = randint[i]
        axes[i][0].imshow(train_X[idx].reshape((size, size)), cmap='gray')
        axes[i][1].imshow(X_reconstruct[idx].reshape((size, size)), cmap='gray')
    plt.savefig('./compare_PCA_reconstruct.png')
    plt.show()


def predict(train_Z, test_Z, train_y, test_y, k=3):
    '''
    train_Z : ndarray (N, 25)
    test_Z : ndarray (N, 25)
    k : k in k-nn
    return -> accuracy
    '''
    # k-nn
    predict_y = np.zeros(test_y.shape[0])
    for i in range(len(test_y)):
        dist = np.zeros(train_Z.shape[0])
        for j in range(train_Z.shape[0]):
            dist[j] = np.sum(np.square(test_Z[i, :] - train_Z[j, :]))
        sort_index = np.argsort(dist)
        neighbors_y = train_y[sort_index[:k]]
        unique, counts = np.unique(neighbors_y, return_counts=True)
        neighbors_y = [k for k, v in sorted(dict(zip(unique, counts)).items(), key=lambda item: -item[1])]
        predict_y[i] = neighbors_y[0]
    
    acc = np.sum(((test_y - predict_y) == 0)) / len(test_y)
    return acc


if __name__ == '__main__':
    # resize (size, size)
    size = 50
    train_X, train_y = read_img('./Yale_Face_Database/Training', size)

    ######################################################################
    # Part 1
    ######################################################################
    # k largest eigenvalues
    k = 25
    eigenvalues, eigenvectors, X_mean = PCA(train_X, k)
    show_eigenfaces(eigenvectors, k, size)

    # reduce dimension (projection) (N, k)
    train_Z = (train_X - X_mean).dot(eigenvectors)

    # reconstruct
    X_reconstruct = train_Z.dot(eigenvectors.T) + X_mean
    show_reconstruct(train_X, X_reconstruct, size)

    ######################################################################
    # Part 2
    ######################################################################
    num_neighbors = 3
    test_X, test_y = read_img('./Yale_Face_Database/Testing', size)
    X = test_X - X_mean
    test_Z = X.dot(eigenvectors)
    acc = predict(train_Z, test_Z, train_y, test_y, num_neighbors)
    print(f'PCA accuracy : {acc * 100:.4f}%')


    ######################################################################
    # Part 3
    ######################################################################
    eigenvalues, eigenvectors, K_center = kernel_PCA(train_X, k)

    train_Z = K_center.dot(eigenvectors)

    num_neighbors = 3
    test_Z = project_kernel_PCA(eigenvectors, train_X, test_X)
    acc = predict(train_Z, test_Z, train_y, test_y, num_neighbors)
    print(f'kernel PCA accuracy : {acc * 100:.4f}%')