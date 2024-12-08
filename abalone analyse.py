'''
    README: remove the "#" and "''' '''" in the function to get the lists where to calculate the mean 
    define 4 functions to get different data which meets the 4 fifferent reqiurements
    respectively: linear_model with all features 
                  linear_model with 2 most correlated features
                  logistic_model with all features
                  logistic_model with 2 most correlated features
'''


from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import seaborn as sns
 
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from math import *
import random
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from keras.optimizers import Adam
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

#log
def read_data_log(r,normalise=False):
    data_in = np.genfromtxt("data/abalone.data", delimiter=",", dtype=str)

    for i in range(len(data_in)):
        if data_in[i][0] == 'M':
            data_in[i][0] = 0
        elif data_in[i][0] == 'F':
            data_in[i][0] = 1
        elif data_in[i][0] == 'I':
            data_in[i][0] = 2

    data_in = data_in.astype(float)

    data_inputx = data_in[:,0:8]
    data_inputy = data_in[:,-1]
    '''classification about age: above 7 and below 7'''
    data_inputy = (data_inputy >= 7).astype(int)
    if normalise:
        transformer = Normalizer().fit(data_inputx)  
        data_inputx = transformer.transform(data_inputx)  

    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.40, random_state=r)
    return x_train, x_test, y_train, y_test,data_in

#2 features logistic regression
def read_data_log_2f(r,normalise=False):
    data_in = np.genfromtxt("data/abalone.data", delimiter=",", dtype=str)

    for i in range(len(data_in)):
        if data_in[i][0] == 'M':
            data_in[i][0] = 0
        elif data_in[i][0] == 'F':
            data_in[i][0] = 1
        elif data_in[i][0] == 'I':
            data_in[i][0] = 2

    data_in = data_in.astype(float)

    data_inputx = data_in[:,[2,7]]
    data_inputy = data_in[:,-1]
    '''classification about age: above 7 and below 7'''
    data_inputy = (data_inputy >= 7).astype(int)
    if normalise:
        transformer = Normalizer().fit(data_inputx)  
        data_inputx = transformer.transform(data_inputx)  

    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.40, random_state=r)
    return x_train, x_test, y_train, y_test,data_in

#linear model read
def read_data(r,normalise=False):
    data_in = np.genfromtxt("data/abalone.data", delimiter=",", dtype=str)

    for i in range(len(data_in)):
        if data_in[i][0] == 'M':
            data_in[i][0] = 0
        elif data_in[i][0] == 'F':
            data_in[i][0] = 1
        elif data_in[i][0] == 'I':
            data_in[i][0] = 2

    data_in = data_in.astype(float)

    data_inputx = data_in[:,0:8]
    data_inputy = data_in[:,-1]
    if normalise:
        transformer = Normalizer().fit(data_inputx)  
        data_inputx = transformer.transform(data_inputx)  

    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.40, random_state=r)
    return x_train, x_test, y_train, y_test,data_in

#linear_model with 2 features
def read_data_l2(r,normalise=False):
    data_in = np.genfromtxt("data/abalone.data", delimiter=",", dtype=str)

    for i in range(len(data_in)):
        if data_in[i][0] == 'M':
            data_in[i][0] = 0
        elif data_in[i][0] == 'F':
            data_in[i][0] = 1
        elif data_in[i][0] == 'I':
            data_in[i][0] = 2

    data_in = data_in.astype(float)

    data_inputx = data_in[:,[2,7]]
    data_inputy = data_in[:,-1]
    if normalise:
        transformer = Normalizer().fit(data_inputx)  
        data_inputx = transformer.transform(data_inputx)  

    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.40, random_state=r)
    return x_train, x_test, y_train, y_test,data_in

def plot_scatter_shellweight_Diameter(data_in):
    df = pd.DataFrame(data_in, columns=["Sex", "Length", "Diameter", "Height", "Whole Weight", "Shucked Weight", "Viscera Weight", "Shell Weight", "ring_age"])

    # Scatter plot: Shell Weight vs Ring Age
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Shell Weight"], df["ring_age"], color='b', alpha=0.5)
    plt.scatter(df["Diameter"], df["ring_age"], color='orange', alpha=0.5)
    plt.title("Shell Weight and Diameter vs Ring Age")
    plt.xlabel("Values")
    plt.ylabel("Ring Age")
    plt.grid(True)
    plt.savefig("scatter_shellweight and Diameter")
    plt.show()

def plot_sex_vs_ring_age(data_in):
    df = pd.DataFrame(data_in, columns=["Sex", "Length", "Diameter", "Height", "Whole Weight", "Shucked Weight", "Viscera Weight", "Shell Weight", "ring_age"])

    # Scatter plot: Sex vs Ring Age (Sex is categorical, so it can be represented by discrete values)
    plt.figure(figsize=(8, 6))
    plt.scatter(df["Sex"], df["ring_age"], color='r', alpha=0.5)
    plt.title("Sex vs Ring Age")
    plt.xlabel("Sex (0: M, 1: F, 2: I)")
    plt.ylabel("Ring Age")
    plt.xticks([0, 1, 2], ["Female", "Male", "Infant"])  # Labeling the categories for better readability
    plt.grid(True)
    plt.savefig("scatter_sex")
    plt.show()

def plot_histograms(data_in):
    df = pd.DataFrame(data_in, columns=["Sex", "Length", "Diameter", "Height", "Whole Weight", "Shucked Weight", "Viscera Weight", "Shell Weight", "ring_age"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
   
    axes[0].hist(df["Shell Weight"], bins=25, color='b', alpha=0.7)
    axes[0].set_title("Distrubution of Shell weight")
    axes[0].set_xlabel("shell_weight")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True)

    axes[1].hist(df["Diameter"], bins=25, color='orange', alpha=0.7)
    axes[1].set_title("Distribution of Diameter")
    axes[1].set_xlabel("Diameter")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True)

    axes[2].hist(df["ring_age"], bins=25, color='pink', alpha=0.7)
    axes[2].set_title("Distribution of ring_age")
    axes[2].set_xlabel("ring age")
    axes[2].set_ylabel("Frequency")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("histograms")
      
def linear_mod(x_train, x_test, y_train, y_test,RMSE,R2,expn):
    print('{}:Running linear regression model...'.format(expn+1))
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)

    y_pred = regr.predict(x_test)

    '''print('Coefficients: \n', regr.coef_)#weight'''   
    RMSE.append(round(sqrt(mean_squared_error(y_test, y_pred)),3))
    R2.append(round(r2_score(y_test, y_pred),3))

    '''print("R Mean squared error (RMSE): %.3f" % sqrt(mean_squared_error(y_test, y_pred)))
    print('Variance score (R²): %.3f' % r2_score(y_test, y_pred))
    print()'''
    return RMSE,R2

def linear_mod_2f(x_train, x_test, y_train, y_test,RMSE_,R2_,expn):
    print('{}:Running linear regression model(2 features)...'.format(expn+1))
    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)

    y_pred = regr.predict(x_test)

    '''print('Coefficients: \n', regr.coef_)#weight'''   
    RMSE_.append(round(sqrt(mean_squared_error(y_test, y_pred)),3))
    R2_.append(round(r2_score(y_test, y_pred),3))

    '''print("R Mean squared error (RMSE): %.3f" % sqrt(mean_squared_error(y_test, y_pred)))
    print('Variance score (R²): %.3f' % r2_score(y_test, y_pred))
    print()'''
    return RMSE_,R2_

def train_linear(RMSE, R2):
    for i in range(30):
        x_train, x_test, y_train, y_test, data_in = read_data(i, normalise=True)  # use True and False to turn on/off the normalisation
        regr = linear_model.LinearRegression()
        regr.fit(x_train, y_train)

        y_pred = regr.predict(x_test)  

        residuals = y_test - y_pred
        
        RMSE.append(round(sqrt(mean_squared_error(y_test, y_pred)), 3))
        R2.append(round(r2_score(y_test, y_pred), 3))

        
        if i == 15:  
            plt.figure(figsize=(8, 6))
            plt.scatter(y_pred, residuals, color='blue', alpha=0.5)
            plt.axhline(y=0, color='orange', linestyle='--', linewidth=2)  
            plt.title(f'Residual Plot for Linear Regression (Final Experiment)')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals (Actual - Predicted)')
            plt.grid(True)
            plt.savefig(f'residual_plot(all features).png')
            plt.show()
    
    return RMSE, R2

def train_linear_2f(RMSE_,R2_):#2 features
    for i in range(30):
        x_train, x_test, y_train, y_test,data_in= read_data_l2(i,normalise = True)#use True and False to turn on/off the normalisation
        linear_mod_2f(x_train, x_test, y_train, y_test,RMSE_,R2_,i)
    return RMSE_,R2_

def mean(RMSE,R2):
    totM = 0
    totR = 0
    for i in range(len(RMSE)):
        totM += RMSE[i]
        totR += R2[i]
    RMSE_mean = round(totM/len(RMSE),3)
    R2_mean = round(totR/len(RMSE),3)
    return RMSE_mean,R2_mean

def mean_(RMSE_,R2_):# for two features
    totM = 0
    totR = 0
    for i in range(len(RMSE_)):
        totM += RMSE_[i]
        totR += R2_[i]
    RMSE_Mean = round(totM/len(RMSE_),3)
    R2_Mean = round(totR/len(RMSE_),3)
    return RMSE_Mean,R2_Mean

def logistic_mod(x_train, x_test, y_train, y_test,ACC,AUCROC,LL,F):
    
    model = LogisticRegression(penalty='l2', C=20.0, solver='liblinear') ####################### 
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]  

    f1 = round(f1_score(y_test, y_pred),3)
    accuracy = round(accuracy_score(y_test, y_pred),3)
    auc_roc = round(roc_auc_score(y_test, y_prob),3)
    log_loss_val = round(log_loss(y_test, y_prob),3)

    AUCROC.append(auc_roc)
    LL.append(log_loss_val)
    ACC.append(accuracy)
    F.append(f1)
    cm = confusion_matrix(y_test, y_pred)
    '''print()
    print(cm)
    print()

    print(f'Accuracy: {accuracy:.3f}')
    print(f'AUC-ROC: {auc_roc:.3f}')
    print(f'Log Loss: {log_loss_val:.3f}')
    print(f'F1 score: {f1:.3f}')'''
    
    return accuracy, auc_roc, log_loss_val,cm

def logistic_mod(x_train, x_test, y_train, y_test,ACC_,AUCROC_,LL_,F_):
    
    model = LogisticRegression(penalty='l2', C=20.0, solver='liblinear') ####################### 
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]  

    f1_ = round(f1_score(y_test, y_pred),3)
    accuracy_ = round(accuracy_score(y_test, y_pred),3)
    auc_roc_ = round(roc_auc_score(y_test, y_prob),3)
    log_loss_val_ = round(log_loss(y_test, y_prob),3)

    AUCROC_.append(auc_roc_)
    LL_.append(log_loss_val_)
    ACC_.append(accuracy_)
    F_.append(f1_)
    cm = confusion_matrix(y_test, y_pred)
    '''print()
    print(cm)
    print()

    print(f'Accuracy: {accuracy:.3f}')
    print(f'AUC-ROC: {auc_roc:.3f}')
    print(f'Log Loss: {log_loss_val:.3f}')
    print(f'F1 score: {f1:.3f}')'''
    
    return accuracy_, auc_roc_, log_loss_val_,cm

def train_logistic(ACC, AUCROC, LL, F):
    last_fpr = None
    last_tpr = None
    last_auc_roc = None
    for i in range(30):
        print("{}:Training the logistic model".format(i+1))
        x_train, x_test, y_train, y_test, data_in = read_data_log(i, normalise=False)  ###############################
        accuracy, auc_roc, log_loss_val, cm = logistic_mod(x_train, x_test, y_train, y_test, ACC, AUCROC, LL, F)
               
        if i == 15:  
            y_prob = LogisticRegression(penalty='l1', C=1.0, solver='liblinear').fit(x_train, y_train).predict_proba(x_test)[:, 1]
            last_fpr, last_tpr, _ = roc_curve(y_test, y_prob)
            last_auc_roc = auc_roc
            cm = cm
        #print()

    return ACC, AUCROC, LL, F, last_fpr, last_tpr, last_auc_roc,cm

def train_logistic_2f(ACC_, AUCROC_, LL_, F_):
    for i in range(30):
        print("{}:Training the logistic model with 2 features".format(i+1))
        x_train, x_test, y_train, y_test, data_in = read_data_log_2f(i, normalise=False)  ###############################
        accuracy_, auc_roc_, log_loss_val_, cm = logistic_mod(x_train, x_test, y_train, y_test, ACC_, AUCROC_, LL_, F_)
    if i == 29:
        cm_ = cm
    return ACC_, AUCROC_, LL_, F_,cm_

def Mean(ACC,AUCROC,LL,F):
    acc = 0
    aucroc = 0
    logloss = 0
    f1 = 0
    tota = 0
    totauc = 0
    totll = 0
    totf = 0
    for i in range(len(LL)):
        tota += ACC[i]
        totauc += AUCROC[i]
        totll += LL[i]
        totf += F[i]
    acc = round(tota/len(LL),3)
    aucroc = round(totauc/len(LL),3)
    logloss = round(totll/len(LL),3)
    f1 = round(totf/len(LL),3)
    return acc,aucroc,logloss,f1

def Mean_(ACC_,AUCROC_,LL_,F_):
    acc = 0
    aucroc = 0
    logloss = 0
    f1 = 0
    tota = 0
    totauc = 0
    totll = 0
    totf = 0
    for i in range(len(LL_)):
        tota += ACC_[i]
        totauc += AUCROC_[i]
        totll += LL_[i]
        totf += F_[i]
    acc_ = round(tota/len(LL_),3)
    aucroc_ = round(totauc/len(LL_),3)
    logloss_ = round(totll/len(LL_),3)
    f1_ = round(totf/len(LL_),3)
    return acc_,aucroc_,logloss_,f1_

def free(RMSE,R2):
    RMSE = []
    R2 = []
    return RMSE,R2

def frees(ACC,AUCROC,LL,F):
    ACC = []
    AUCROC = []
    LL = []
    F=[]
    return ACC,AUCROC,LL,F

def nn():
    data_in = np.genfromtxt("data/abalone.data", delimiter=",", dtype=str)

    for i in range(len(data_in)):
        if data_in[i][0] == 'M':
            data_in[i][0] = 0
        elif data_in[i][0] == 'F':
            data_in[i][0] = 1
        elif data_in[i][0] == 'I':
            data_in[i][0] = 2

    data_in = data_in.astype(float)

    data_inputx = data_in[:,0:8]
    data_inputy = data_in[:,-1]
    data_inputy = (data_inputy >= 7).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.4, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    print("Standard Deviation of each feature after scaling:", scaler.scale_)

    model = Sequential()
    model.add(Dense(32, input_dim=x_train.shape[1], activation='relu'))  
    model.add(Dense(16, activation='relu'))  
    model.add(Dense(1, activation='sigmoid'))  

    sgd = SGD(learning_rate=0.1)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(x_test, y_test))

    # 评估模型
    model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(x_test, y_test))

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'The Accuracy of nn is: {accuracy:.4f}')
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('DenseNet Result')
    plt.show()
    return x_train,y_train,x_test,y_test

def mlp_model():
    data_in = np.genfromtxt("data/abalone.data", delimiter=",", dtype=str)

    for i in range(len(data_in)):
        if data_in[i][0] == 'M':
            data_in[i][0] = 0
        elif data_in[i][0] == 'F':
            data_in[i][0] = 1
        elif data_in[i][0] == 'I':
            data_in[i][0] = 2

    data_in = data_in.astype(float)

    data_inputx = data_in[:, 0:8]  # 使用前 8 列作为特征
    data_inputy = data_in[:, -1]   

    # 对目标进行二值化：环数大于等于7为1，否则为0（分类任务）
    data_inputy = (data_inputy >= 7).astype(int)

    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.4, random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = Sequential()
    model.add(Dense(64, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu')) 
    model.add(Dense(1, activation='sigmoid'))  

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), verbose=1)

    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Final Test Accuracy: {accuracy:.4f}')

    y_pred = model.predict(x_test)
    y_pred_classes = (y_pred >= 0.5).astype(int)  

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))

    rmse = mean_squared_error(y_test, y_pred_classes, squared=False)
    print(f'RMSE: {rmse:.4f}')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig("MLP result")
    plt.show()

    return model

def calculate_std(RMSE, R2, RMSE_, R2_, ACC, AUCROC, LL, F, ACC_, AUCROC_, LL_, F_):
    # Linear model with all features
    RMSE_std = round(np.std(RMSE), 3)
    R2_std = round(np.std(R2), 3)
    print(f"Linear Model with all features - RMSE std: {RMSE_std}, R² std: {R2_std}")

    # Linear model with 2 features
    RMSE__std = round(np.std(RMSE_), 3)
    R2__std = round(np.std(R2_), 3)
    print(f"Linear Model with 2 features - RMSE std: {RMSE__std}, R² std: {R2__std}")

    # Logistic model with all features
    ACC_std = round(np.std(ACC), 3)
    AUCROC_std = round(np.std(AUCROC), 3)
    LL_std = round(np.std(LL), 3)
    F_std = round(np.std(F), 3)
    print(f"Logistic Model with all features - Accuracy std: {ACC_std}, AUC-ROC std: {AUCROC_std}, Log Loss std: {LL_std}, F1 Score std: {F_std}")

    # Logistic model with 2 features
    ACC__std = round(np.std(ACC_), 3)
    AUCROC__std = round(np.std(AUCROC_), 3)
    LL__std = round(np.std(LL_), 3)
    F__std = round(np.std(F_), 3)
    print(f"Logistic Model with 2 features - Accuracy std: {ACC__std}, AUC-ROC std: {AUCROC__std}, Log Loss std: {LL__std}, F1 Score std: {F__std}")
def main():
    r = 0
    expn = 0
    RMSE  =[]
    R2 = []

    RMSE_ = []
    R2_ = []

    ACC = []
    AUCROC = []
    LL = []
    F = []
    
    ACC_ = []
    AUCROC_ = []
    LL_ = []
    F_ = []

    x_train, x_test, y_train, y_test, data_in = read_data(r)
    
    '''print(x_train, ' x_train')
    print(y_train, ' y_train')'''

    '''df is important!!!!!!!!'''
    df = pd.DataFrame(data_in[:, :9], columns=["Sex", "Length", "Diameter", "Height", "Whole Weight", "Shucked Weight", "Viscera Weight", "Shell Weight", "ring_age"])
    corr_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")

    plt.savefig("correlation_heatmap")
    plt.show()
    '''^^^ plot heatmap'''

    plot_scatter_shellweight_Diameter(data_in)#2 positive correlations
    plot_sex_vs_ring_age(data_in)#1 negative Correlation
    plot_histograms(data_in)#2 most positive correlations distributions compare with the distribution of ring age

    '''calculate the mean of RMSE AND R2'''
    train_linear(RMSE,R2)#train 30 experiments
    train_linear_2f(RMSE_,R2_)
    RMSE_mean,R2_mean = mean(RMSE,R2)
    RMSE_Mean,R2_Mean = mean_(RMSE_,R2_)

    free(RMSE,R2)
    
    ACC, AUCROC, LL, F, last_fpr, last_tpr, last_auc_roc ,cm= train_logistic(ACC, AUCROC, LL, F)
    ACC_, AUCROC_, LL_, F_ ,cm_= train_logistic_2f(ACC_, AUCROC_, LL_, F_)
    acc,aucroc,logloss,f1 = Mean(ACC,AUCROC,LL,F)
    acc_,aucroc_,logloss_,f1_ = Mean_(ACC_,AUCROC_,LL_,F_)
    
    '''ROC PLOT'''
    plt.figure()
    plt.plot(last_fpr, last_tpr, color='blue', label=f'ROC curve (AUC = {last_auc_roc:.3f})')
    plt.plot([0, 1], [0, 1], color='pink', linestyle='--')  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig("ROC_curve.png")  
    plt.show()

    '''about linear model'''
    #remove "#" to get the list of the results
    
    
    print()
    x_train,x_test,y_train,y_test = nn()
    model = mlp_model()
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("The result of linear model with all features:")
    #print("RMSE list is :")
    #print(RMSE)
    print("RMSE mean is {}".format(RMSE_mean))
    #print("R2 list is :")
    #print(R2)
    print("R2 mean is {}".format(R2_mean))
    print()


    '''about logistic model'''
    print("The result of logistic model with all features:")
    #print("The Accuracy List is:")
    #print(ACC)
    print("The accuracy mean is {}".format(acc))
    #print("The AUC-ROC List is:")
    #print(AUCROC)
    print("The auc_roc mean is {}".format(aucroc))
    #print("The log loss value list is:")
    #print(LL)
    print("The mean of log_loss_val is {}".format(logloss))
    #print("The f1 score list is:")
    #print(F)
    print("The mean of f1 score is {}".format(f1))
    
    print("The confusion_matrix is:")
    print(cm)
    print()

    
    '''linear model with 2 features'''
    print("The result of linear model with 2 features:")
    #print("RMSE list is :")
    #print(RMSE_)
    print("RMSE mean is {}".format(RMSE_Mean))
    #print("R2 list is :")
    #print(R2_)
    print("R2 mean is {}".format(R2_Mean))
    print()


    '''about logistic model with 2 features'''
    print("The result of logistic model with 2 features:")
    #print("The Accuracy List is:")
    #print(ACC_)
    print("The accuracy mean is {}".format(acc_))
    #print("The AUC-ROC List is:")
    #print(AUCROC_)
    print("The auc_roc mean is {}".format(aucroc_))
    #print("The log loss value list is:")
    #print(LL_)
    print("The mean of log_loss_val is {}".format(logloss_))
    #print("The f1 score list is:")
    #print(F_)
    print("The mean of f1 score is {}".format(f1_))
    print("The confusion_matrix is:")
    print(cm_)
    print()
    calculate_std(RMSE, R2, RMSE_, R2_, ACC, AUCROC, LL, F, ACC_, AUCROC_, LL_, F_)
    
    
    frees(ACC,AUCROC,LL,F)
    free(RMSE,R2)

if __name__ == '__main__':
     main()



