import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
from collections import Counter
import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)


def neural_net(output_units, input_units, hidden_units):
    def build_NNmodel():
        model = Sequential()
        model.add(Dense(units = hidden_units, activation='sigmoid', input_dim=input_units))
        model.add(Dense(units=output_units, activation='softmax'))
        # optimize model with stochastic gradient descent with categorical_crossentropy loss function
        model.compile(loss='categorical_crossentropy', optimizer= 'sgd', metrics=['accuracy'])
        return model
    clf = KerasClassifier(build_fn = build_NNmodel, epochs = 300,  verbose = 0)
    return clf

def cross_validation(df, filename):
    # classification using decision tree, random forest, and logistic regression
    y = df['OS_MONTHS']
    columns = list(df)
    excluded_attributes = ['PATIENT_ID', 'SAMPLE_ID', 'OS_STATUS', 'OS_MONTHS', 'VITAL_STATUS']
    X = df.drop(columns = excluded_attributes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    output_units = len(set(y))
    input_units = len(list(X))
    hidden_units = input_units
    # convert y_train and y_test to binary vector using one_hot_encoding
    nn_X_test = X_test.values
    nn_y_test = y_test.values
    nn_y_test = np_utils.to_categorical(nn_y_test)
    nn_X_train = X_train.values
    nn_y_train = y_train.values
    nn_y_train = np_utils.to_categorical(nn_y_train)

    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    logReg = LogisticRegression()
    nn = neural_net(output_units, input_units, hidden_units)
    clfs = {'DecisionTree': dt, 'RandomForest': rf, 'LogisticRegression': logReg, 'NeuralNet': nn}
    results = None
    for classifier in clfs.keys():
        clf = clfs[classifier]
        if classifier != 'NeuralNet':
            scores = cross_val_score(clf, X_train, y_train, cv = 5, scoring = 'accuracy')
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, pred)
        
        else:
            scores = cross_val_score(clf, nn_X_train, nn_y_train, cv = 5)
            clf.fit(nn_X_train, nn_y_train)
            pred = clf.predict(nn_X_test)
            cm = confusion_matrix(y_test, pred)
        print('Average and sdev of accuracy scores of training {} is: {} and {}'.format(classifier,scores.mean(), scores.std()), file = filename)
        print('Confusion matrix of {} is: \n {}'.format(classifier, cm), file = filename)
        print('Predicted accuracy of {} is {}'.format(classifier, accuracy_score(y_test, pred)), file = filename)
        if classifier == 'NeuralNet':
            results = [pred,y_test]
    return results

def main(args):
    folder = args.Data
    filename = args.out
    dataSets = ['clinical_data.csv', 'clinical_GE.csv', 'clinical_CNA.csv', 'clinical_GE_CNA.csv']
    #dataSets = ['clinical_GE_CNA.csv']
    with open(os.path.join(folder,filename), 'w') as f:
        for i in range(0, len(dataSets)):
            df = pd.read_csv(os.path.join(folder, dataSets[i]))
            X_train, X_test, y_train, y_test = train_test_split(df, df['OS_MONTHS'], test_size = 0.25, random_state = 0)
            name = dataSets[i]
            name = name.replace('.csv', '')
            X_test.to_csv(os.path.join(folder, name+'_testSet.csv'), index = False)
            print('Class distribution in train and test sets of  {} is: {}'.format(dataSets[i], Counter(y_train), Counter(y_test)), file = f)
            # perform cross validation
            results = cross_validation(df,f)
            if dataSets[i] == 'clinical_GE_CNA.csv' and results != None:
                pred = np.asarray(results[0])
                actual = np.asarray(results[1])
                for j in range(0, len(pred)):
                    print(pred[j], '\t', actual[j], file = f)

    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--Data', type=str, required=True,
                                   help='name of data folder')
    parser.add_argument('--out', type=str, required=True,
                                   help='name of output file')
    args = parser.parse_args()

    main(args)


