import numpy as np
from scipy.spatial.distance import squareform, pdist
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
    return ->
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


def LDA (X, y, dim):
    '''
    X : ndarray (N, D) , N (number of data), D (dimension of image)
    y : ndarray (N), the label of datas
    dim : first dim largest eigenvalues 
    return ->
        eigenvalues : ndarray (dim)
        eigenvectors : ndarray (D, dim)
    '''
    N, D = X.shape
    X_mean = np.mean(X, axis=0).reshape(1, -1)
    class_means = np.zeros((15, D)) 

    for i in range(N):
        class_means[train_y[i]] += X[i]
    class_means = class_means / 9 

    # within-class scatter
    S_W = np.zeros((D, D))
    for i in range(N):
        diff = X[i, :].reshape(-1, 1) - class_means[y[i]].reshape(-1, 1)
        S_W = S_W + diff.dot(diff.T)

    # between-class scatter
    S_B = np.zeros((D, D))
    for i in range(15):
        diff = class_means[i, :].reshape(-1, 1) - X_mean.reshape(-1, 1)
        S_B = S_B + 9 * diff.dot(diff.T)

    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    sort_index = np.argsort(-eigenvalues)
    sort_index = sort_index[:dim]
    eigenvalues = np.real(eigenvalues[sort_index])
    eigenvectors = np.real(eigenvectors[:, sort_index])

    return eigenvalues, eigenvectors


def kernel_LDA (X, dim):
    '''
    X : ndarray (N, D) , N (number of data), D (dimension of image)
    dim : first dim largest eigenvalues 
    return ->
        eigenvalues : ndarray (dim)
        eigenvectors : ndarray (D, dim)
        K : Gram matrix
    '''
    (N, D) = X.shape
    # kernel 1 (RBF)
    gamma = 0.00000006
    K1 = squareform(np.exp(-gamma * pdist(X, 'sqeuclidean')))
    # kernel 2 (polynomial)
    K2 = np.zeros((N, N))
    alpha, c, poly = 2, 1, 3
    for i in range(N):
        for j in range(N):
            K2[i, j] = (alpha * X[i, :].reshape(1, -1).dot(X[j, :].reshape(-1, 1)) + c) ** poly

    K = K2

    # within-class scatter
    S_W = np.zeros((N, N))
    ones_Nr = np.ones((9, 9)) / 9
    for i in range(15):
        S_W += K[:, i * 9:(i+1) * 9] @ (np.identity(9) - ones_Nr) @ K[:, i * 9:(i+1) * 9].T

    # between-class scatter
    S_B = np.zeros((N, N))
    for r in range(15):
        M_r = np.sum(K[:, r * 9:(r+1) * 9], axis=1)
        middle_term = np.zeros((N, N))
        for p in range(15):
            middle_term += np.sum(K[:, p * 9:(p+1) * 9], axis=1).dot(M_r.T)
        for q in range(15):
            middle_term += M_r.dot(np.sum(K[:, q * 9:(q+1) * 9], axis=1).T)
        last_term = np.zeros((N, N))
        for p in range(15):
            for q in range(15):
                last_term += np.sum(K[:, p * 9:(p+1) * 9], axis=1).dot(np.sum(K[:, q * 9:(q+1) * 9], axis=1).T)
        S_B += (M_r.dot(M_r.T) - middle_term / 15 + last_term / 15 / 15)

    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    sort_index = np.argsort(-eigenvalues)
    sort_index = sort_index[:dim]
    eigenvalues = np.real(eigenvalues[sort_index])
    eigenvectors = np.real(eigenvectors[:, sort_index])

    return eigenvalues, eigenvectors, K


def project_kernel_LDA (eigenvectors, train_X, test_X):
    '''
    eigenvectors : ndarray (N, dim)
    train_X : ndarray (N, D) , N (number of training data), D (dimension of image)
    test_X : ndarray (N, D) , N (number of testing data), D (dimension of image)
    return -> ndarray (N, dim), N (number of testing data)
    '''
    X = np.vstack((train_X, test_X))
    (n, d) = X.shape
    # kernel 1 (RBF)
    gamma = 0.00000006

    K1 = squareform(np.exp(-gamma * pdist(X, 'sqeuclidean')))
    # kernel 2 (polynomial)
    K2 = np.zeros((n, n))
    alpha, c, poly = 2, 1, 3
    for i in range(n):
        for j in range(n):
            K2[i, j] = (alpha * X[i, :].reshape(1, -1).dot(X[j, :].reshape(-1, 1)) + c) ** poly

    N = train_X.shape[0]
    K = K2[N:, :N].copy()

    return K.dot(eigenvectors)


def show_fisherfaces(eigenvectors, k=25, size=25):
    eigenfaces = eigenvectors.T.reshape((k, size, size))

    fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(8, 10))
    for i in range(k):
        axes[i // 5][i % 5].set_title(f'{i + 1}')
        axes[i // 5][i % 5].imshow(eigenfaces[i], cmap="gray")
    plt.savefig('./first_25_fisherfaces.png')
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
    plt.savefig('./compare_LDA_reconstruct.png')
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

    eigenvalues_pca, eigenvectors_pca, X_mean = PCA(train_X, dim=31)
    X_pca = (train_X - X_mean).dot(eigenvectors_pca)
    eigenvalues_lda, eigenvectors_lda = LDA(X_pca, train_y, k)

    eigenvectors = eigenvectors_pca.dot(eigenvectors_lda)

    show_fisherfaces(eigenvectors, k, size)

    # reduce dimension (projection) (N, k)
    train_Z = train_X.dot(eigenvectors)

    # reconstruct
    X_reconstruct = train_Z.dot(eigenvectors.T)
    show_reconstruct(train_X, X_reconstruct, size)

    ######################################################################
    # Part 2
    ######################################################################
    num_neighbors = 3
    test_X, test_y = read_img('./Yale_Face_Database/Testing', size)
    test_Z = test_X.dot(eigenvectors)
    acc = predict(train_Z, test_Z, train_y, test_y, num_neighbors)
    print(f'LDA accuracy : {acc * 100:.4f}%')

    ######################################################################
    # Part 3
    ######################################################################
    eigenvalues_lda, eigenvectors_lda, K = kernel_LDA(X_pca, k)
    train_Z = K.dot(eigenvectors_lda)

    test_X_pca = (test_X - X_mean).dot(eigenvectors_pca)

    num_neighbors = 3
    test_Z = project_kernel_LDA(eigenvectors_lda, X_pca, test_X_pca)
    acc = predict(train_Z, test_Z, train_y, test_y, num_neighbors)
    print(f'kernel LDA accuracy : {acc * 100:.4f}%')
