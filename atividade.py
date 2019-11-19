import cv2
import glob
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from timeit import default_timer


algorithms = ['kNN3', 'kNN5', 'kNN7', 'WkNN3', 'WkNN5', 'WkNN7', 'NB',
              'DT', 'SVMLinear', 'SVMrbf', 'LR', 'NN']


def values_for_dataframe(values, columns):
    return pd.DataFrame([list(i) for i in zip(*values)], columns=columns)
    

def feature_extractor(arrayimgs):
    squarescarac = []

    for x in arrayimgs:

        aux = []

        _, c, _ = cv2.findContours(x, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # peri = cv2.arcLength(c[0], True)  # perimetro
        # aux.append(peri)
        #
        # aproxx = cv2.approxPolyDP(c[0], 0.04 * peri, True)  # vertices
        # vertc = len(aproxx)
        # aux.append(vertc)
        #
        # area = cv2.contourArea(c[0])  # area
        # aux.append(area)

        momentum = cv2.moments(x)   # centroide
        # cX = int(momentum["m10"] / momentum["m00"])
        # cY = int(momentum["m01"] / momentum["m00"])
        #
        # aux.append(cX)
        # aux.append(cY)

        moments = cv2.HuMoments(momentum, True).flatten()

        for m in moments:
            aux.append(m)

        squarescarac.append(aux)

    return squarescarac


def turntogray(arrayphotos):
    size = len(arrayphotos)
    for x in range(0, size):
        arrayphotos[x] = cv2.cvtColor(arrayphotos[x], cv2.COLOR_BGR2GRAY)

    return arrayphotos


def reshape2dto1d(arrayphotos):
    size = len(arrayphotos)
    for x in range(0, size):
        arrayphotos[x] = arrayphotos[x].ravel()

    return arrayphotos


def resizephotos(arrayphotos, size1, size2):
    size = len(arrayphotos)
    for x in range(0, size):
        arrayphotos[x] = cv2.resize(arrayphotos[x], (size1, size2))

    return arrayphotos


def gaussianblurArray(arrayphotos, val1, val2):
    for x in range(len(arrayphotos)):
        arrayphotos[x] = cv2.GaussianBlur(arrayphotos[x], (val1, val1), val2)

    return arrayphotos


def binaryadaptive(arrayphotos, threshold, val1):
    for x in range(len(arrayphotos)):
        arrayphotos[x] = cv2.adaptiveThreshold(arrayphotos[x], threshold, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                               val1, 10)

    return arrayphotos


def invertbinaryphotos(arrayphotos):
    for x in range(len(arrayphotos)):
        arrayphotos[x] = cv2.bitwise_not(arrayphotos[x])

    return arrayphotos


def loadfolderimgs(arrayphotos, path):
    for img in glob.glob(path):
        n = cv2.imread(img)
        arrayphotos.append(n)

    return arrayphotos


def reshape3dto2d(arrayphotos):
    size = len(arrayphotos)
    for x in range(0, size):
        arrayphotos[x] = np.reshape(arrayphotos[x], (arrayphotos[x].shape[0], (arrayphotos[x].shape[1]*arrayphotos[x].shape[2])))


def return_label(type):
    if 'circle' in type:
        return 0
    elif 'rectangle' in type:
        return 1
    elif 'ellipse' in type:
        return 2
    elif 'hexagon' in type:
        return 3
    elif 'line' in type:
        return 4
    elif 'rhombus' in type:
        return 5
    elif 'square' in type:
        return 6
    elif 'trapezium' in type:
        return 7
    elif 'triangle' in type:
        return 8


def loadfolderimgs(path):
    labels = []
    arrayphotos = []
    for img in glob.glob(path):
        n = cv2.imread(img)
        labels.append(return_label(img))
        arrayphotos.append(n)

    return arrayphotos, pd.DataFrame(labels, columns=['label'])


def reconhecimento(base, splits=10, iteration=5):
    kf = KFold(n_splits=splits, shuffle=True)

    knn3 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn5 = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn7 = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
    w_knn3 = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')
    w_knn5 = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
    w_knn7 = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
    nb = GaussianNB()
    dt = DecisionTreeClassifier()
    svm_linear = svm.SVC(kernel='linear')
    svm_rbf = svm.SVC(gamma='scale', kernel='rbf')
    lr = LogisticRegression(solver='lbfgs')
    nn = MLPClassifier()

    knn3_time = []
    knn3_accuracy = []
    knn5_time = []
    knn5_accuracy = []
    knn7_time = []
    knn7_accuracy = []
    w_knn3_time = []
    w_knn3_accuracy = []
    w_knn5_time = []
    w_knn5_accuracy = []
    w_knn7_time = []
    w_knn7_accuracy = []
    nb_time = []
    nb_accuracy = []
    dt_time = []
    dt_accuracy = []
    svm_linear_time = []
    svm_linear_accuracy = []
    svm_rbf_time = []
    svm_rbf_accuracy = []
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
            knn3.fit(train_data, train_label)
            predict_label = knn3.predict(test_data)
            fim = default_timer()
            knn3_time.append(fim - inicio)
            knn3_accuracy.append(accuracy_score(test_label, predict_label))

            inicio = default_timer()
            knn5.fit(train_data, train_label)
            predict_label = knn5.predict(test_data)
            fim = default_timer()
            knn5_time.append(fim - inicio)
            knn5_accuracy.append(accuracy_score(test_label, predict_label))

            inicio = default_timer()
            knn7.fit(train_data, train_label)
            predict_label = knn7.predict(test_data)
            fim = default_timer()
            knn7_time.append(fim - inicio)
            knn7_accuracy.append(accuracy_score(test_label, predict_label))

            inicio = default_timer()
            w_knn3.fit(train_data, train_label)
            predict_label = w_knn3.predict(test_data)
            fim = default_timer()
            w_knn3_time.append(fim - inicio)
            w_knn3_accuracy.append(accuracy_score(test_label, predict_label))

            inicio = default_timer()
            w_knn5.fit(train_data, train_label)
            predict_label = w_knn5.predict(test_data)
            fim = default_timer()
            w_knn5_time.append(fim - inicio)
            w_knn5_accuracy.append(accuracy_score(test_label, predict_label))

            inicio = default_timer()
            w_knn7.fit(train_data, train_label)
            predict_label = w_knn7.predict(test_data)
            fim = default_timer()
            w_knn7_time.append(fim - inicio)
            w_knn7_accuracy.append(accuracy_score(test_label, predict_label))

            inicio = default_timer()
            nb.fit(train_data, train_label)
            predict_label = nb.predict(test_data)
            fim = default_timer()
            nb_time.append(fim - inicio)
            nb_accuracy.append(accuracy_score(test_label, predict_label))

            inicio = default_timer()
            dt.fit(train_data, train_label)
            predict_label = dt.predict(test_data)
            fim = default_timer()
            dt_time.append(fim - inicio)
            dt_accuracy.append(accuracy_score(test_label, predict_label))

            inicio = default_timer()
            svm_linear.fit(train_data, train_label)
            predict_label = svm_linear.predict(test_data)
            fim = default_timer()
            svm_linear_time.append(fim - inicio)
            svm_linear_accuracy.append(accuracy_score(test_label, predict_label))

            inicio = default_timer()
            svm_rbf.fit(train_data, train_label)
            predict_label = svm_rbf.predict(test_data)
            fim = default_timer()
            svm_rbf_time.append(fim - inicio)
            svm_rbf_accuracy.append(accuracy_score(test_label, predict_label))

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

    accuracy = values_for_dataframe([knn3_accuracy, knn5_accuracy, knn7_accuracy, w_knn3_accuracy, 
                                     w_knn5_accuracy, w_knn7_accuracy, nb_accuracy, dt_accuracy,
                                     svm_linear_accuracy, svm_rbf_accuracy, lr_accuracy, nn_accuracy],
                                    algorithms)

    time = values_for_dataframe([knn3_time, knn5_time, knn7_time, w_knn3_time,
                                 w_knn5_time, w_knn7_time, nb_time, dt_time,
                                 svm_linear_time, svm_rbf_time, lr_time, nn_time],
                                algorithms)
    return accuracy, time


def normalize(df1):
    x = df1.values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(scaled)
    return df_normalized


data, label = loadfolderimgs('2d_geometric_shapes_dataset/*.jpg')
shapes = turntogray(data)
shapes = gaussianblurArray(shapes, 27, 0)
# shapes = binaryadaptive(shapes, 255, 37)
# shapes = invertbinaryphotos(shapes)
# shapes_features = feature_extractor(shapes)
# data_set = pd.DataFrame(shapes_features)
# data_set = normalize(data_set)
# data_set = data_set.join(label)
# data_set = data_set.sample(frac=1)
#
# acc, times = reconhecimento(data_set)
# acc.to_csv('acuracias_norm.csv', index=False)
# times.to_csv('tempos_norm.csv', index=False)

# cv2.drawContours(clone, cnts, -1, (0,255,0),2)
# print("found " + str(len(cnts)) + " countours")
# cv2.imshow("all contours", clone)
# cv2.waitKey(0)


acc = pd.read_csv('acuracias_norm.csv')
times = pd.read_csv('tempos_norm.csv')

for a in algorithms:
    print(a)
    print('media - ', np.mean(acc[a]))
    print('mediana - ', np.median(acc[a]))
    print('desvio padrão - ', np.std(acc[a]))
    print('================')
