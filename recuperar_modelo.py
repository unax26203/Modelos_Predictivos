import getopt
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

model = "best_model.pkl"
p = "./"

if __name__ == '__main__':
    print('ARGV   :', sys.argv[1:])
    try:
        options, remainder = getopt.getopt(sys.argv[1:], 'p:m:f:h', ['path=', 'model=', 'testFile=', 'h'])
    except getopt.GetoptError as err:
        print('ERROR:', err)
        sys.exit(1)
    print('OPTIONS   :', options)

    for opt, arg in options:
        if opt in ('-p', '--path'):
            p = arg
        elif opt in ('-f', '--file'):
            f = arg
        elif opt in ('-m', '--model'):
            m = arg
        elif opt in ('-h', '--help'):
            print(' -p modelAndTestFilePath \n -m modelFileName -f testFileName\n ')
            exit(1)

    if p == './':
        model = p + str(m)
        iFile = p + str(f)
    else:
        model = p + "/" + str(m)
        iFile = p + "/" + str(f)

    y_test = pd.DataFrame()
    testX = pd.read_csv(iFile)

    # Eliminar la columna 'TARGET' de los datos de prueba
    if 'TARGET' in testX.columns:
        testX.drop(columns=['TARGET'], inplace=True)

    print(testX.head(5))
    clf = pickle.load(open(model, 'rb'))
    predictions = clf.predict(testX)
    probas = clf.predict_proba(testX)
    y_test['preds'] = predictions
    predictions = pd.Series(data=predictions, index=testX.index, name='predicted_value')
    results_test = testX.join(predictions, how='left')

    print(results_test)

    # Obtener informe de clasificación detallado
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # Obtener matriz de confusión
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
