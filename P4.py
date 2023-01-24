import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import pair_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class validation_set:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

class test_set:
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

class data_set:
    def __init__(self, validation_set, test_set):
        self.validation_set = validation_set
        self.test_set = test_set

def generate_train_test(file_name):
    
        pd.options.display.max_colwidth = 200
    
        #Usando el dataset de iris pasar a un dataframe de pandas y dividirlo en train y test

        df = pd.read_csv(file_name, sep=',', engine='python')
        X = df.drop(['species'],axis=1).values
        y = df['species'].values
    
        #Separa el corpus cargado en el DataFrame en el 70% para entrenamiento y el 30% para pruebas

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 0)
    
        #~ #Crea pliegues para la validación cruzada
        validation_sets = []
        kf = KFold(n_splits=3)
        for train_index, test_index in kf.split(X_train):
    
            #print("TRAIN:", train_index, "\n",  "TEST:", test_index)
            X_train_, X_test_ = X_train[train_index], X_train[test_index]
            y_train_, y_test_ = y_train[train_index], y_train[test_index]
    
            validation_sets.append(validation_set(X_train_, y_train_, X_test_, y_test_))
    
    
        #~ #Crea el conjunto de test
        test_set_ = test_set(X_test, y_test)

        #~ #Crea el conjunto de datos
        mydata_set_ = data_set(validation_sets, test_set_)

        #return mydata_set_
        return mydata_set_


def avg(list) :
    return sum(list)/len(list)




if __name__ == "__main__":
    
    #Genera el conjunto de datos
    mydata_set = generate_train_test("iris.csv")

    #usando la funcion de gradiente estocastico crearemos una linea que se ajuste a los datos de entrenamiento y validacion cruzada, e imprimiremos los resultados de la funcion de perdida y el coeficiente de determinacion para cada pliegue de validacion cruzada.

    i=0
    gauss_sum_acc = []
    mult_sum_acc = []

    #para cada uno de los pliegues de validacion cruzada
    for validation_set in mydata_set.validation_set:
        i+=1
        print("Pliegue de validacion cruzada: ", i)
        #print(validation_set.X_train)
        #print(validation_set.y_train)
        #print(validation_set.X_test)
        #print(validation_set.y_test)
        #print("")

        #Entrena el modelo
        gauss_model = GaussianNB()
        gauss_model.fit(validation_set.X_train, validation_set.y_train)
        mult_model = MultinomialNB()
        mult_model.fit(validation_set.X_train, validation_set.y_train)

        #Predice con el modelo
        gauss_y_pred = gauss_model.predict(validation_set.X_test)
        mult_y_pred = mult_model.predict(validation_set.X_test)

        #Evalua el modelo
        gauss_acc = accuracy_score(validation_set.y_test, gauss_y_pred)
        mult_acc = accuracy_score(validation_set.y_test, mult_y_pred)

        gauss_sum_acc.append(gauss_acc)
        mult_sum_acc.append(mult_acc)

        print("GaussianNB Accuracy: ", gauss_acc)
        print("MultinomialNB Accuracy: ", mult_acc)
        print("")

    print("GaussianNB Accuracy promedio: ", avg(gauss_sum_acc))
    print("MultinomialNB Accuracy promedio: ", avg(mult_sum_acc))

##Usando el mejor metodo de clasificacion que fue el de GaussianNB, usando todos los datos sin pliegues de validacion cruzada, se obtiene el siguiente resultado:

#generar train y test
    df = pd.read_csv('iris.csv', sep=',', engine='python')
    X = df.drop(['species'],axis=1).values
    y = df['species'].values

    #Separa el corpus cargado en el DataFrame en el 70% para entrenamiento y el 30% para pruebas

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle = True, random_state = 0)

    #Entrena el modelo
    gauss_model = GaussianNB()
    gauss_model.fit(X_train, y_train)

    #Predice con el modelo
    gauss_y_pred = gauss_model.predict(X_test)

    #Evalua el modelo
    gauss_acc = accuracy_score(y_test, gauss_y_pred)

    print("GaussianNB Accuracy: ", gauss_acc)

#obtener la metrica de precision

    print("Precision: ", precision_score(y_test, gauss_y_pred, average='macro'))

#obtener la metrica de recall

    print("Recall: ", recall_score(y_test, gauss_y_pred, average='macro'))

#obtener la metrica de f1

    print("F1: ", f1_score(y_test, gauss_y_pred, average='macro'))

#obtener la metrica de confusion matrix

    print("Confusion Matrix:\n ", confusion_matrix(y_test, gauss_y_pred))

#plot confusion matrix






