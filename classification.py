import cv2
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from timeit import default_timer
import glob
import numpy as np
import pickle

algorithms = ['LR', 'NN']

path = 'DatasetOrganizado/organizado/'


def normalize(df1):
    x = df1.values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(scaled)
    return df_normalized


def loadfolderimgs():
    labels = []
    arrayphotos = []
    for i in range(10):
        for img in glob.glob(path + "/" + str(i) + "/*"):
            arrayphotos.append(cv2.imread(img))
            labels.append(i)

    return arrayphotos, pd.DataFrame(labels, columns=['label'])


def values_for_dataframe(values, columns):
    return pd.DataFrame([list(i) for i in zip(*values)], columns=columns)


def extract_GB(img):
    grayscale = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    kernel = cv2.getGaborKernel(ksize=(64, 64), sigma=0.02, theta=0, lambd=1, gamma=0.02, psi=0)
    afterGB = cv2.filter2D(grayscale, -1, kernel)
    features = cv2.resize(afterGB, (8, 8))
    return features.flatten()


def feature_extractor(arrayimgs):
    squarescarac = []

    for x in arrayimgs:
        feat = extract_GB(x)
        squarescarac.append(feat)

    return squarescarac


def reconhecimento(base, splits=10, iteration=5):
    kf = KFold(n_splits=splits, shuffle=True)

    lr = LogisticRegression(solver='lbfgs')
    nn = MLPClassifier()

    lr_time = []
    lr_accuracy = []
    nn_time = []
    nn_accuracy = []

    for iter in range(1, iteration+1):
        print('iteração: ', iter)
        data = base.copy()
        labels = data.pop('label')

        fold = 1
        for train_index, test_index in kf.split(base):
            print('fold: ', fold)
            train_data, test_data = data.iloc[train_index], data.iloc[test_index]
            train_label, test_label = labels.iloc[train_index], labels.iloc[test_index]

            inicio = default_timer()
            lr.fit(train_data, train_label)
            predict_label = lr.predict(test_data)
            fim = default_timer()
            lr_time.append(fim - inicio)
            lr_accuracy.append(accuracy_score(test_label, predict_label))

            inicio = default_timer()
            nn.fit(train_data, train_label)
            predict_label = nn.predict(test_data)
            fim = default_timer()
            nn_time.append(fim - inicio)
            nn_accuracy.append(accuracy_score(test_label, predict_label))

            fold += 1
        base = base.sample(frac=1)

    accuracy = values_for_dataframe([lr_accuracy, nn_accuracy], algorithms)

    time = values_for_dataframe([lr_time, nn_time], algorithms)
    return accuracy, time


def create_dataset():
    imgs, label = loadfolderimgs()
    data = feature_extractor(imgs)
    data = pd.DataFrame(data)
    dataset = data.join(label)
    dataset = dataset.sample(frac=1)

    return dataset


def run():
    dataset = create_dataset()
    acc, times = reconhecimento(dataset)
    acc.to_csv('acuracias.csv', index=False)
    times.to_csv('tempos.csv', index=False)

    for a in algorithms:
        print(a)
        print('media - ', np.mean(acc[a]))
        print('mediana - ', np.median(acc[a]))
        print('desvio padrão - ', np.std(acc[a]))
        print('================')


def treinar_modelo():
    dataset = create_dataset()
    lr = LogisticRegression(solver='lbfgs')
    labels = dataset.pop('label')
    lr.fit(dataset, labels)
    arq = open("model_lr.md", 'wb')
    pickle.dump(lr, arq)
    arq.close()


treinar_modelo()