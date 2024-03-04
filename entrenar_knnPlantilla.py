# -*- coding: utf-8 -*-
from getopt import getopt
from sys import exit, argv, version_info
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import datetime
from sklearn.preprocessing import LabelEncoder
import pickle

# Variables globales
OUTPUT_FILE = "output.txt"
INPUT_FILE = "SantanderTraHalfHalf.csv"
TARGET_NAME = "TARGET"
K_MIN = 1
K_MAX = 3
D = ['uniform', 'distance']
P_MIN = 1
P_MAX = 2
DEV_SIZE = 0.2
RANDOM_STATE = 42
ALGORITHMS = ['knn', 'decision_tree', 'random_forest']  # Lista de algoritmos disponibles
SELECTED_ALGORITHM = 'knn'  # Algoritmo seleccionado por defecto
N_ESTIMATORS = 100
CRITERION = 'gini'

train = None
dev = None
best_model = None
best_f1_score = 0


def usage():
    print("Usage: entrenar_knn.py <optional-args>")
    print("The options supported by entrenar_knn are:")
    print(f"-a              algorithm to use ->                      DEFAULT: {SELECTED_ALGORITHM}")
    print(f"-d              distance parameter ->                   DEFAULT: {D}")
    print(f"-h, --help      show the usage")
    print(f"-i, --input     input file path of the data            DEFAULT: ./{INPUT_FILE}")
    print(f"-k-min          number of neighbors for the KNN algorithm   DEFAULT: {K_MIN}")
    print(f"-k-max          number of neighbors for the KNN algorithm   DEFAULT: {K_MAX}")
    print(f"-o, --output    output file path for the results            DEFAULT: ./{OUTPUT_FILE}")
    print(f"-p-min          distance from -> 1: Manhatan | 2: Euclidean DEFAULT: {P_MIN}")
    print(f"-p-max          distance to -> 1: Manhatan | 2: Euclidean   DEFAULT: {P_MIN}")


def load_options(options):
    global SELECTED_ALGORITHM, INPUT_FILE, OUTPUT_FILE, K_MIN, K_MAX, D, P_MIN, P_MAX

    for opt, arg in options:
        if opt == "-d":
            D = arg.split(',')
        elif opt in ('-h', '--help'):
            usage()
            exit(0)
        elif opt in ('-i', '--input'):
            INPUT_FILE = str(arg)
        elif opt == '-k-min':
            K_MIN = int(arg)
        elif opt == '-k-max':
            K_MAX = int(arg)
        elif opt in ('-o', '--output'):
            OUTPUT_FILE = str(arg)
        elif opt == '-p-min':
            P_MIN = int(arg)
        elif opt == '-p-max':
            P_MAX = int(arg)
        elif opt == '-a':
            if arg.lower() not in ALGORITHMS:
                print("Error: Invalid algorithm specified.")
                usage()
                exit(1)
            SELECTED_ALGORITHM = arg.lower()


def show_script_options():
    print("entrenar_knn.py configuration:")
    print(f"-a algorithm to use        -> {SELECTED_ALGORITHM}")
    print(f"-d distance parameter      -> {D}")
    print(f"-i input file path         -> {INPUT_FILE}")
    print(f"-k number of neighbors     -> from: {K_MIN} to: {K_MAX}")
    print(f"-o output file path        -> {OUTPUT_FILE}")
    print(f"-p distance algorithm      -> from: {P_MIN} to: {P_MAX}")


def coerce_to_unicode(x):
    if version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x, 'utf-8')
        else:
            return unicode(x)

    return str(x)


def atributos_excepto(atributos, excepciones):
    atribs = []

    for a in atributos:
        if a not in excepciones:
            atribs.append(a)

    return atribs


def imprimir_atributos(atributos):
    string = ""
    for atr in atributos:
        string += str(f"{atr} ")
    print("---- Atributos seleccionados")
    print(string)
    print()


def datetime_to_epoch(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    delta = dt - epoch
    return delta.total_seconds()


def estandarizar_tipos_de_datos(dataset, categorical_features, numerical_features, text_features):
    label_encoders = {}
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        dataset[feature] = label_encoders[feature].fit_transform(dataset[feature].astype(str))

    for feature in text_features:
        dataset[feature] = dataset[feature].apply(coerce_to_unicode)

    for feature in numerical_features:
        if dataset[feature].dtype == np.dtype('M8[ns]') or (
                hasattr(dataset[feature].dtype, 'base') and dataset[feature].dtype.base == np.dtype('M8[ns]')):
            dataset[feature] = datetime_to_epoch(dataset[feature])
        else:
            try:
                dataset[feature] = dataset[feature].astype('double')
            except ValueError:
                print(f"Could not convert feature '{feature}' to double.")


def obtener_lista_impute_para(atributos, impute_with, excepciones):
    lista = []
    for a in atributos:
        if a not in excepciones:
            entrada = {"feature": a, "impute_with": impute_with}
            lista.append(entrada)

    return lista


def obtener_lista_rescalado_para(atributos, rescale_with, excepciones):
    diccionario = {}
    for a in atributos:
        if a not in excepciones:
            diccionario[a] = rescale_with;

    return diccionario


def preprocesar_datos(dataset, drop_rows_when_missing, impute_when_missing, rescale_features):
    train, dev = train_test_split(dataset, test_size=DEV_SIZE, random_state=RANDOM_STATE,
                                  stratify=dataset['__target__'])

    if drop_rows_when_missing:
        train.dropna(inplace=True)
        dev.dropna(inplace=True)

    if impute_when_missing:
        for feature in impute_when_missing:
            if feature['impute_with'] == 'MEAN':
                v1 = train[feature['feature']].mean()
                v2 = dev[feature['feature']].mean()
            elif feature['impute_with'] == 'MEDIAN':
                v1 = train[feature['feature']].median()
                v2 = dev[feature['feature']].median()
            elif feature['impute_with'] == 'MODE':
                v1 = train[feature['feature']].value_counts().index[0]
                v2 = dev[feature['feature']].value_counts().index[0]
            elif feature['impute_with'] == 'CONSTANT':
                v1 = feature['value']
                v2 = v1
            train[feature['feature']] = train[feature['feature']].fillna(v1)
            dev[feature['feature']] = dev[feature['feature']].fillna(v2)

            s1 = f"- Train feature {feature['feature']} with value {str(v1)}"
            s2 = f"- Dev feature {feature['feature']} with value {str(v2)}"
            print("Imputed missing values\t%s\t%s" % (s1, s2))

    for (feature_name, rescale_method) in rescale_features.items():
        if rescale_method == 'MINMAX':
            min_val = min(train[feature_name].min(), dev[feature_name].min())
            max_val = max(train[feature_name].max(), dev[feature_name].max())
            train[feature_name] = (train[feature_name] - min_val) / (max_val - min_val)
            dev[feature_name] = (dev[feature_name] - min_val) / (max_val - min_val)
        elif rescale_method == 'ZSCORE':
            mean = train[feature_name].mean()
            std = train[feature_name].std()
            train[feature_name] = (train[feature_name] - mean) / std
            dev[feature_name] = (dev[feature_name] - mean) / std

    return train, dev


def comprobar_modelo(modelo, devX, devY, target_map, output_file, k, p, w):
    global best_model, best_f1_score

    predictions = modelo.predict(devX)
    f1 = f1_score(devY, predictions, average='weighted')

    if f1 > best_f1_score:
        best_f1_score = f1
        best_model = modelo
        with open(output_file, 'a') as file:
            file.write(f"Best model updated: Algorithm: {SELECTED_ALGORITHM} | k: {k} p:{p} w:{w}\n")
            file.write("Model evaluation:\n")
            file.write(classification_report(devY, predictions) + "\n")
            file.write("Confusion matrix:\n")
            file.write(str(confusion_matrix(devY, predictions, labels=[1, 0])) + "\n\n")

    print(f"Algorithm: {SELECTED_ALGORITHM} | k: {k} p:{p} w:{w}")
    print("Results have been saved to:", output_file)
    print(f1_score(devY, predictions, average=None))
    print(classification_report(devY, predictions))
    print(confusion_matrix(devY, predictions, labels=[1, 0]))


def aplicar_undersampling(trainX, trainY):
    undersampler = RandomUnderSampler()
    trainX_resampled, trainY_resampled = undersampler.fit_resample(trainX, trainY)
    return trainX_resampled, trainY_resampled


def aplicar_oversampling(trainX, trainY):
    oversampler = RandomOverSampler()
    trainX_resampled, trainY_resampled = oversampler.fit_resample(trainX, trainY)
    return trainX_resampled, trainY_resampled


def main():
    global dev, train, best_model

    # Limpiar el archivo de salida antes de empezar
    with open(OUTPUT_FILE, 'w') as f:
        f.write("")

    print("---- Iniciando main...")

    ml_dataset = pd.read_csv(INPUT_FILE)

    atributos = ml_dataset.columns
    imprimir_atributos(atributos)

    ml_dataset = ml_dataset[atributos]

    print("---- Estandarizamos en Unicode y pasamos de atributos categoricos a numericos")
    categorical_features = []
    text_features = []
    numerical_features = atributos_excepto(ml_dataset.columns, [TARGET_NAME] + categorical_features + text_features)
    estandarizar_tipos_de_datos(ml_dataset, categorical_features, numerical_features, text_features)

    print("---- Tratamos el TARGET: " + TARGET_NAME)
    target_map = {'0': 0, '1': 1}
    ml_dataset_copy = ml_dataset.copy()
    ml_dataset_copy['__target__'] = ml_dataset_copy[TARGET_NAME].map(str).map(target_map)
    del ml_dataset_copy[TARGET_NAME]
    ml_dataset_copy = ml_dataset_copy[~ml_dataset_copy['__target__'].isnull()]

    print("---- Dataset empleado")
    print(ml_dataset.head(5))

    drop_rows_when_missing = False
    impute_when_missing = []
    rescale_features = {}

    print("---- Preprocesamos los datos")
    train, dev = preprocesar_datos(ml_dataset_copy, drop_rows_when_missing, impute_when_missing, rescale_features)

    print("---- Dataset preprocesado")
    print("TRAIN: ")
    print(train.head(5))
    print(train['__target__'].value_counts())
    print("DEV: ")
    print(dev.head(5))
    print(dev['__target__'].value_counts())

    trainX = train.drop('__target__', axis=1)
    devX = dev.drop('__target__', axis=1)
    trainY = np.array(train['__target__'])
    devY = np.array(dev['__target__'])

    print("---- Aplicando undersampling")
    trainX_resampled, trainY_resampled = aplicar_undersampling(trainX, trainY)

    print("---- Iniciando barrido de parámetros ")
    output_file = 'output.txt'
    for k in range(K_MIN, K_MAX + 1):
        for p in range(P_MIN, P_MAX + 1):
            for w in D:
                print(f"Algorithm: {SELECTED_ALGORITHM} | k: {k} p:{p} w:{w}")
                if SELECTED_ALGORITHM == 'knn':
                    clf = KNeighborsClassifier(n_neighbors=k, weights=w, algorithm='auto', leaf_size=30, p=p)
                elif SELECTED_ALGORITHM == 'decision_tree':
                    clf = DecisionTreeClassifier()
                elif SELECTED_ALGORITHM == 'random_forest':
                    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, criterion=CRITERION)
                else:
                    print("Error: Invalid algorithm selected.")
                    exit(1)

                clf.class_weight = "balanced"
                clf.fit(trainX_resampled, trainY_resampled)
                comprobar_modelo(clf, devX, devY, target_map, output_file, k, p, w)

    print("---- Guardando el mejor modelo...")
    with open('best_model.pkl', 'wb') as model_file:
        pickle.dump(best_model, model_file)

    print("Best model saved as 'best_model.pkl'")


if __name__ == "__main__":
    try:
        options, remainder = getopt(argv[1:], 'a:d:h:i:k-min:k-max:o:p-min:p-max', ['help', 'input', 'output'])
    except getopt.GetoptError as err:
        print("ERROR: ", err)
        exit(1)

    load_options(options)
    show_script_options()
    main()
