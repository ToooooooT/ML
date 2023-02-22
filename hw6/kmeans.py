import numpy as np
from scipy.spatial.distance import squareform, pdist
from PIL import Image
from array2gif import write_gif
import time

def initalize(Gram, k, type):
    # mean of k clusters
    cluster_means = np.zeros((k, Gram.shape[1]))

    if type == 'kmeans++':
        # random select 1 points be center
        cluster_means[0] = Gram[np.random.randint(low=0, high=10000, size=1), :]
        # select k - 1 cluster center
        for c in range(1, k):
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


def compute_gram(image, gamma_s, gamma_c):
    S = np.zeros((10000, 2))
    for i in range(10000):
        S[i] = [i // 100, i % 100]
    # gram = squareform(np.exp(-gamma_s * pdist(S, 'sqeuclidean'))) * squareform(np.exp(-gamma_c * pdist(image, 'sqeuclidean')))
    gram = squareform(np.exp(-gamma_c * pdist(image, 'sqeuclidean')))
    return gram


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

    return gifs


def save_gif (gifs, path):
    for i in range(len(gifs)):
        gifs[i] = gifs[i].transpose(1, 0, 2)
    write_gif(gifs, path, fps=2)


if __name__ == '__main__':
    start = time.time()
    # hyperparameter
    gamma_s1, gamma_c1 = 0.001, 0.001
    gamma_s2, gamma_c2 = 0.001, 0.001

    # number of clusters
    k = 4

    # Load Data
    image1 = Image.open('./image1.png')
    image2 = Image.open('./image2.png')
    image1 = np.array(image1)
    image2 = np.array(image2)

    image1 = image1.reshape((10000, 3))
    image2 = image2.reshape((10000, 3))

    # compute gram matrix
    gram1 = compute_gram(image1, gamma_s1, gamma_c1)
    gram2 = compute_gram(image2, gamma_s2, gamma_c2)

    # initialize method
    init_type='kmeans++'
    # init_type='random'

    # initialize indicator variables
    cluster_means1 = initalize(gram1, k, init_type)
    cluster_means2 = initalize(gram2, k, init_type)

    # k means
    gifs1 = kmeans(cluster_means1, k, gram1)
    gifs2 = kmeans(cluster_means2, k, gram2)

    # save_gif
    save_gif(gifs1, f'./GIF/{k}_cluster_kmeans_{init_type}_SpaceKernel_image1.gif')
    save_gif(gifs2, f'./GIF/{k}_cluster_kmeans_{init_type}_SpaceKernel_image2.gif')

    end = time.time()

    print(f'Execution time : {end - start} sec')
    

    