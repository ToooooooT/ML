import os
import struct
import numpy as np
import math
from tqdm import tqdm
from numba import jit


def load_mnist(path, tag='train'):
    label_path = os.path.join(path, '%s-labels.idx1-ubyte'%tag)
    image_path = os.path.join(path, '%s-images.idx3-ubyte'%tag)
    with open(label_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(image_path, 'rb') as imgpath:
        magic, n, row, col = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), row*col)
    return labels, images

@jit(nopython=True)
def AssignLabel(X, labels, LAMBDA, P):
    counting = np.zeros((10, 10), dtype=np.int32)
    mapping = np.zeros((10), dtype=np.int32)
    for i in range(train_num):
        p = np.zeros((10), dtype=np.float64)
        for j in range(10):
            p_ = LAMBDA[j]
            for k in range(pixels):
                p_ *= (P[j, k] ** X[i, k])
                p_ *= ((1 - P[j, k]) ** (1 - X[i, k]))
            p[j] = p_
        counting[labels[i], np.argmax(p)] += 1
    for i in range(10):
        idx = np.argmax(counting)
        label = idx // 10 
        class_idx = idx % 10 
        mapping[label] = class_idx
        counting[label, :] = -1
        counting[:, class_idx] = -1
    return mapping


def PrintClass(P, mapping, labeled):
    for i in range(10):
        if labeled:
            print('labeled ', end='')
        print('class: ' + str(i))
        class_idx = mapping[i]
        for r in range(img_size):
            for c in range(img_size):
                char = '1' if P[class_idx, r * img_size + c] >= 0.5 else ' '
                print(char, end=' ')
            print() # 換行
        print() # 換行


@jit(nopython=True)
def CalculateCofusionMatrix(X, labels, LAMBDA, P, mapping):
    mapping_inverse = np.zeros((10), dtype=np.int32)
    for i in range(10):
        # 將mapping反過來，原本mapping是哪個label對應到哪個class，這裡用成哪個class對應到哪個label
        mapping_inverse[i] = np.where(mapping == i)[0][0]
    confusion_matrix = np.zeros((10, 2, 2))
    for i in range(train_num):
        p = np.zeros((10), dtype=np.float64)
        for j in range(10):
            p_ = LAMBDA[j]
            for k in range(pixels):
                p_ *= (P[j, k] ** X[i, k])
                p_ *= ((1 - P[j, k]) ** (1 - X[i, k]))
            p[j] = p_
        prediction = mapping_inverse[np.argmax(p)]
        for j in range(10):
            if labels[i] == j:
                if prediction == j:
                    confusion_matrix[j][0][0] += 1
                else:
                    confusion_matrix[j][0][1] += 1
            else:
                if prediction == j:
                    confusion_matrix[j][1][0] += 1
                else:
                    confusion_matrix[j][1][1] += 1
    return confusion_matrix

# @jit(nopython=True)
def PrintResult(X, labels, LAMBDA, P, mapping, iteration):
    confusion_matrix = CalculateCofusionMatrix(X, labels, LAMBDA, P, mapping)

    for i in range(10):
        print('---------------------------------------------------------------\n')
        print(f'Confusion Matrix {i} :')
        print(f'\t\tPredict number {i} Predict not number {i}')
        print(f'Is number {i}\t\t{confusion_matrix[i][0][0]}\t\t{confusion_matrix[i][0][1]}')
        print(f'Isn\'t number {i}\t\t{confusion_matrix[i][1][0]}\t\t{confusion_matrix[i][1][1]}')
        sens = confusion_matrix[i][0][0] / (confusion_matrix[i][0][0] + confusion_matrix[i][0][1])
        spec = confusion_matrix[i][1][1] / (confusion_matrix[i][1][0] + confusion_matrix[i][1][1])
        print(f'\nSensitivity (Successfully predict number {i})\t: {sens}')
        print(f'Specificity (Successfully predict not number {i})\t: {spec}\n')
    
    error = train_num - np.sum(confusion_matrix[:, 0, 0])
    print(f'Total iteration to converge: {iteration}')
    print(f'Total error rate: {error / train_num}')

@jit(nopython=True)
def E_Step(train_num, class_num, pixels, train_imgs, lamb, p):
    w = np.zeros((train_num, class_num))
    # for i in tqdm(range(train_num)):
    for i in range(train_num):
        for j in range(class_num):
            w[i][j] = lamb[j]
            for k in range(pixels):
                w[i][j] *= (p[j][k]**train_imgs[i][k])
                w[i][j] *= ((1 - p[j][k])**(1-train_imgs[i][k]))
        if(np.sum(w[i]) == 0):
             continue
        w[i] = w[i]/np.sum(w[i])
    return w

@jit(nopython=True)
def M_Step(train_num, class_num, pixels, train_imgs, lamb, p, w):
    for i in range(class_num):

        for j in range(pixels):
            w_sum = 0
            p[i, j] = 0.
            for k in range(train_num):
                p[i, j] += w[k, i]*train_imgs[k, j]
                w_sum += w[k, i]
            if(w_sum != 0):
                p[i, j] /= w_sum
            lamb[i] = w_sum / train_num


if __name__ == '__main__':
    # 0~9
    class_num = 10
    # 這次只需要用到training data
    train_lbs, train_imgs =  load_mnist('./data/', tag='train')

    pixels = train_imgs[0].shape[0]
    img_size = int(math.sqrt(pixels))
    train_num = train_lbs.shape[0]
    # train_num = 60000

    # 初始機率 -> 選到0~9的機率
    lamb = np.full((class_num), 1/class_num)
    # 初始機率 -> 對應的class，在某個pixel選擇1的機率
    p = np.random.uniform(0.0, 1.0, (class_num, pixels))

    print('prepare data:')
    for i in tqdm(range(train_num)):
        for j in range(pixels):
            if (train_imgs[i][j] >= 128):
                train_imgs[i][j] = 1
            else:
                train_imgs[i][j] = 0

    print('Start:')
    mapping = np.array([i for i in range(10)])
    count = 0
    while (True):
        count += 1

        # E Step
        w = E_Step(train_num, class_num, pixels, train_imgs, lamb, p)
        '''
        w = np.zeros((train_num, class_num))
        for i in tqdm(range(train_num)):
            for j in range(class_num):
                w[i][j] = lamb[j]
                for k in range(pixels):
                    w[i][j] *= (p[j][k]**train_imgs[i][k])
                    w[i][j] *= ((1 - p[j][k])**(1-train_imgs[i][k]))
            if(sum(w[i]) == 0):
                continue
            w[i] = w[i]/sum(w[i])
        '''
        
        
        # M Step
        p_last = p.copy()
        M_Step(train_num, class_num, pixels, train_imgs, lamb, p, w)
        '''
        for i in range(class_num):
            # w_sum = 0
            # w_sum = sum(w[:, i])
            # lamb = (w_sum/train_num)
            for j in range(pixels):
                w_sum = 0
                p[i, j] = 0.
                for k in range(train_num):
                    p[i, j] += w[k, i]*train_imgs[k, j]
                    w_sum += w[k, i]
                if(w_sum != 0):
                    p[i, j] /= w_sum
                lamb[i] = w_sum / train_num
        '''
        
        delta = sum(sum(abs(p - p_last)))
        PrintClass(p, mapping, False)
        print(f"No. of Iteration: {count}, Difference: {delta}\n")
        print("--------------------------------------------------------")


        # if (delta <= 5 and count >= 10):
        if (delta <= 5 or count >= 20):
            break
    
    print('------------------------------------------------------------\n')
    mapping = AssignLabel(train_imgs, train_lbs, lamb, p)
    PrintClass(p, mapping, True)
    PrintResult(train_imgs, train_lbs, lamb, p, mapping, count)
