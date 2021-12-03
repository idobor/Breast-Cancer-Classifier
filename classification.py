from math import gamma
from os import name
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tqdm

def load_data():
  data=load_breast_cancer()
  return data

def find_best_svc_model(data):
  linear_score = 0
  quard_score = 0
  gausian_score = 0
  linear = SVC(kernel="linear",C=0.1)
  quard = SVC(kernel="poly")
  gausian = SVC(kernel="rbf",gamma=1000)

  for i in tqdm.tqdm(range(100)):
    X_train,X_test,y_train,y_test = train_test_split(data.data,data.target,test_size=0.2)
    linear.fit(X_train,y_train)
    quard.fit(X_train,y_train)
    gausian.fit(X_train,y_train)
    linear_predictions = linear.predict(X_test)
    quard_predictions = quard.predict(X_test)
    gausian_predictions = quard.predict(X_test)
    linear_score += accuracy_score(linear_predictions,y_test)
    quard_score += accuracy_score(quard_predictions,y_test)
    gausian_score += accuracy_score(gausian_predictions,y_test)
  print("Linear Score is: ",linear_score/100)
  print("Quard Score is: ",quard_score/100)
  print("Gausian Score is: ",gausian_score/100)



def find_best_c_for_linear(data):
    C_s= [10**i for i in np.arange(0.0,0.5,0.05)]
    print(C_s)
    scores =[]
    for i in tqdm.tqdm(range(len(C_s))):
      c=C_s[i]
      linear_score=0
      for i in tqdm.tqdm(range(15)):
        X_train,X_test,y_train,y_test = train_test_split(data.data,data.target,test_size=0.2)
        linear = SVC(kernel="linear",C=c)
        linear.fit(X_train,y_train)
        linear_predictions = linear.predict(X_test)
        linear_score += accuracy_score(linear_predictions,y_test)
      scores.append(linear_score/15)
    
    plt.plot(C_s,scores)
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.show()
    print(C_s[np.argmax(scores)])
    print(max(scores))



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    """
    w=np.zeros(len(data[0]))
    eta_t=eta_0
    n=len(data)
    for t in range(T):
      eta_t=eta_0/(t+1)
      i=np.random.randint(n)
      if np.dot(np.dot(labels[i],w),data[i])<1:
        w=np.dot((1-eta_t),w)+eta_t*C*np.dot(labels[i],data[i])
      else:
        w=np.dot((1-eta_t),w)
    return w
def find_eta_with_SGD_hinge_loss(data,T,C):
    
    etas = [10**(i-5) for i in range(11)]
    accuracy_list = []
    for i,eta in enumerate(etas):
        accuracy=0
        for j in range(10):
            X_train,X_test,y_train,y_test = train_test_split(data.data,data.target,test_size=0.2)
            w = SGD_hinge(X_train,y_train ,C,eta,T)
            accuracy += get_accuracy_hinge(w,X_test,y_test)
        accuracy_list.append(accuracy/10)
    best_eta=etas[np.argmax(accuracy_list)]
    plt.xscale('log')
    plt.xlabel("eta")
    plt.ylabel("accuracy")
    plt.plot(etas,accuracy_list)
    print("best eta is:",best_eta)
    print("best accuracy is:",np.max(accuracy_list))
    plt.show()


def get_accuracy_hinge(w,data,labels):
    accuracy=0
    for i,x in enumerate(data):
        if is_correct_hinge(w,x,labels[i]):
            accuracy+=1
    return accuracy/len(data)

def is_correct_hinge(w,x,y):
    return my_sign(np.dot(w,x))==y
def my_sign(x):
    if x>0:
        return 1
    return 0

if __name__ =="__main__":
  data = load_data()
  #X_train,X_test,y_train,y_test = train_test_split(data.data,data.target,test_size=0.2)
  #find_best_svc_model(data)
  #find_best_c_for_linear(data)
  find_eta_with_SGD_hinge_loss(data,1000,100)

