# === libraries === 
import numpy as np
from sklearn.model_selection import train_test_split
from logisticRegression import LogisticRegression
import pandas as pa

# == custom standard ==
# def standard(X):
#     X['Pregnancies'] = (X['Pregnancies'] - np.mean(X['Pregnancies']))/np.std(X['Pregnancies'])
#     X['Glucose'] = (X['Glucose'] - np.mean(X['Glucose']))/np.std(X['Glucose'])
#     X['BloodPressure'] = (X['BloodPressure'] - np.mean(X['BloodPressure']))/np.std(X['BloodPressure'])
#     X['SkinThickness'] = (X['SkinThickness'] - np.mean(X['SkinThickness']))/np.std(X['SkinThickness'])
#     X['Insulin'] = (X['Insulin'] - np.mean(X['Insulin']))/np.std(X['Insulin'])
#     X['BMI'] = (X['BMI'] - np.mean(X['BMI']))/np.std(X['BMI'])
#     X['DiabetesPedigreeFunction'] = (X['DiabetesPedigreeFunction'] - np.mean(X['DiabetesPedigreeFunction']))/np.std(X['DiabetesPedigreeFunction'])
#     X['Age'] = (X['Age'] - np.mean(X['Age']))/np.std(X['Age'])

# == prebuilt standard ==
# from sklearn.preprocessing import StandardScaler
# def standard(X):
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#     return X

# X_learn = standard(X_learn)

# === read the cleaned data === 
learn_d = pa.read_csv("CleanDataTrain.csv")
test_d = pa.read_csv("CleanDataTest.csv")

X_learn = learn_d.iloc[:, 0:8]
Y_learn = learn_d.iloc[:, 8]

X_test = test_d.iloc[:, 0:8]
Y_test = test_d.iloc[:, 8]

# === calling logistic regression to train itself ===
LR = LogisticRegression(lr = 0.0001, n_iters= 100000)
LR.fit(X_learn, Y_learn)

Y_prediction = LR.predict(X_test)

# === Accuracy ===
def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test)/len(y_test)

# result = accuracy(Y_prediction, Y_test)
# print(result)

# === main === 
while(True):
    print("Select an option: \n 1) Evaluation\n 2) Give input\n 3) Exit Program")
    option = int(input())

    if(option == 1):
        acc = accuracy(Y_test, Y_prediction)
        print("Accuracy is:",acc)

    elif(option == 2):
        pregnancies = float(input("Please enter number of pregnancy you had: "))
        glucose = float(input("Please enter your glucose rate ==> mg/dl: "))
        bloodPressure = float(input("Please enter your blood pressure ==> mm/Hg: "))
        skinThickness = float(input("Please enter thickness of your skin ==> (0,99): "))
        insulin = float(input("Please enter insulin level of your blood ==> mm: "))
        bmi = float(input("Please enter you BMI: "))
        diabetesPedigreeFunction = float(input("Please enter Diabetes pedigree function: "))
        age = float(input("Please enter your age: "))

        x_input = [[pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]]
        prob = LR.predict(x_input)
        print("Outcome: ", prob[0])

    elif(option == 3):
        print("Khodafez")
        break
    
