import numpy as np
from scipy.spatial.distance import squareform, pdist
from PIL import Image
from array2gif import write_gif
import matplotlib.pyplot as plt

def compute_similarity_matrix(image, gamma_s, gamma_c):
    S = np.zeros((10000, 2))
    for i in range(10000):
        S[i] = [i // 100, i % 100]
    W = squareform(np.exp(-gamma_s * pdist(S, 'sqeuclidean'))) * squareform(np.exp(-gamma_c * pdist(image, 'sqeuclidean')))
    return W

def initalize(Gram, k, type):
    # mean of k clusters
    cluster_means = np.zeros((k, Gram.shape[1]))

    if type == 'kmeans++':
        # random select 1 points be center
        cluster_means[0] = Gram[np.random.randint(low=0, high=10000, size=1), :]
        # select k - 1 cluster center
        for c in range(1,k):
            dist = np.zeros((10000, c))
            for i in range(len(Gram)):
                for j in range(c):
                    dist[i, j] = np.sqrt(np.sum((Gram[i] - cluster_means[j])**2))
            dist_min = np.min(dist,axis=1)
            sum = np.sum(dist_min)*np.random.rand()
            for i in range(10000):
                sum -= dist_min[i]
                if sum <= 0:
                    cluster_means[c] = Gram[i]
                    break
    else:
        choose = np.random.randint(low=0, high=10000, size=k)
        cluster_means = Gram[choose, :]

    return cluster_means
    

def get_gif (a):
    '''
    a : indicator variables
    '''
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], [255, 255, 255]])

    res = np.zeros((100, 100, 3))
    for i in range(100):
        for j in range(100):
            label = np.argmax(a[:, i * 100 + j])
            res[i, j, :] = colors[label, :]

    return res.astype(np.uint8)


def kmeans (cluster_means, k, gram):
    '''
    cluster_means : (k, 10000) ndarray
    k : number of cluster
    '''

    diff = 1e10
    count = 1
    gifs = []
    while diff > 1e-10:
        # indicator variables
        a = np.zeros((k, 10000))

        # E step
        for i in range(10000):
            dist = []
            for j in range(k):
                dist.append(np.sum((gram[i] - cluster_means[j, :]) ** 2))
            a[np.argmin(dist), i] = 1

        # M step
        new_cluster_means = np.zeros(cluster_means.shape)
        for i in range(10000):
            label = np.argmax(a[:, i])
            new_cluster_means[label] = new_cluster_means[label] + gram[i]

        for i in range(k):
            if np.sum(a[i, :]) > 0:
                new_cluster_means[i] = new_cluster_means[i] / np.sum(a[i, :])

        diff = np.sum((new_cluster_means - cluster_means) ** 2)
        cluster_means = new_cluster_means

        gif = get_gif(a)
        gifs.append(gif)
        print('---------------------------------------------------------------')
        print(f'iteration {count}')
        for i in range(k):
            print(f'# cluster {i} : {int(np.sum(a[i, :]))}')
        print(f'diff : {diff}')

        count += 1

    return a, gifs


def save_gif (gifs, path):
    for i in range(len(gifs)):
        gifs[i] = gifs[i].transpose(1, 0, 2)
    write_gif(gifs, path, fps=2)


def plot_eigenvector(U, a):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    markers = ['o', '^', 's']
    for i, marker in enumerate(markers):
        x = y = z = []
        for j in range(10000):
            if a[i, j] == 1:
                x.append(U[j, 0])
                y.append(U[j, 1])
                z.append(U[j, 2])
        ax.scatter(x, y, z, marker=marker)

    ax.set_xlabel('x')   
    ax.set_xlabel('y')   
    ax.set_xlabel('z') 
    plt.savefig('./eigenvector.png')


def normalized_spectral_clustering(image_path, gamma_s, gamma_c, k, init_type):
    # Load Data
    image = Image.open(image_path)
    image = np.array(image)
    image = image.reshape((10000, 3))

    # similarity matrix
    W = compute_similarity_matrix(image, gamma_s, gamma_c)

    # degree matrix
    D = np.diag(np.sum(W, axis=1))

    # Laplacian
    L = D - W

    eigenvalues, eigenvectors = np.linalg.eig(L)
    idx = np.argsort(eigenvalues)
    U = eigenvectors[:, idx[1:1+k]]

    cluster_means = initalize(U, k, init_type)
    a, gifs = kmeans(cluster_means, k, U)

    # plot eigenvector
    # plot_eigenvector(U, a)

    return gifs


if __name__ == '__main__':
    # hyperparameter
    gamma_s1, gamma_c1 = 0.001, 0.001
    gamma_s2, gamma_c2 = 0.001, 0.001

    # number of clusters
    k = 2

    # initialize method
    init_type='kmeans++'
    # init_type='random'

    gifs1 = normalized_spectral_clustering('./image1.png', gamma_s1, gamma_c1, k, init_type)
    gifs2 = normalized_spectral_clustering('./image2.png', gamma_s2, gamma_c2, k, init_type)

    save_gif(gifs1, f'./GIF/{k}_unnormalized_spectral_clustering_{init_type}_image1.gif')
    save_gif(gifs2, f'./GIF/{k}_unnormalized_spectral_clustering_{init_type}_image2.gif')
