#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import os,csv
import zipfile
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import glob
import keras
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.utils import np_utils
from keras import regularizers
from keras.models import Sequential,load_model
from keras.utils.np_utils import to_categorical
from sklearn import preprocessing
from keras.layers import Dense, Dropout


# In[4]:


StudentData = pd.read_csv(r'C:\Users\sai teja pusarla\Desktop\studentdata1.CSV')
StudentData.school


# In[5]:



grade = []
for i in StudentData['G3'].values:
    if i in range(0,10):
        grade.append('F')
    elif i in range(10,12):
        grade.append('D')
    elif i in range(12,14):
        grade.append('C')
    elif i in range(14,16):
        grade.append('B')
    else:
        grade.append("A")

StudentData_Copy = StudentData
se = pd.Series(grade)
StudentData_Copy['Grade'] = se.values


# In[6]:



StudentData_Copy.head(2)


# In[7]:



StudentData.isnull().sum()


# In[8]:


studentData_without_G3 =  StudentData_Copy.drop(['G3'], axis=1)
studentData_without_G3.head(5)


# In[9]:


Y = studentData_without_G3.filter(["Grade"],axis=1)
#Y
X = studentData_without_G3.drop(['Grade'],axis=1)
X.head(5)

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.15)
X.head(5)


# In[10]:


x_train = pd.get_dummies(xTrain)
x_test = pd.get_dummies(xTest)
y_train  = pd.get_dummies(yTrain)
y_test  = pd.get_dummies(yTest)
y_train.head(5)


# In[12]:


model = Sequential()
model.add(Dense(64, activation='relu',  kernel_regularizer=regularizers.l2(0.001),input_shape = (58,)))
model.add(Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model.add(Dense(5,  kernel_regularizer=regularizers.l2(0.001),activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history  = model.fit(x_train,y_train, epochs = 200, batch_size = 5, validation_data = (x_test,y_test))
model.save('UCI_model_A.h5')


# In[13]:


model_1 = load_model('UCI_model_A.h5')
model_1.summary()
results = model_1.evaluate(x_test,y_test)
print("Accuracy of the Model %.2f%%" % ( results[1]*100))


# In[14]:


#Predict Y values
y_predicted = model_1.predict(x_test)

#Convert y_test dataframe to letter grades
test_Y = []
for i in y_test.values:
    if i[4].round() ==1:
        test_Y.append('F')
    elif i[3].round() ==1:
        test_Y.append('D')
    elif i[2].round() == 1:
        test_Y.append('C')
    elif i[1].round() ==1:
        test_Y.append('B')
    elif i[0].round() ==1:
        test_Y.append('A')
 
#convert y_predicted dataframe to letter grades
predicted_Y1 = []

for i in y_predicted:
    if i[4].round() == 1:
        predicted_Y1.append('F')
    elif i[3].round() ==1:
        predicted_Y1.append('D')
    elif i[2].round() == 1:
        predicted_Y1.append('C')
    elif i[1].round() ==1:
        predicted_Y1.append('B')
    else:
        predicted_Y1.append('A')


# In[15]:


#Code for visualizing confusion Matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False, 
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[16]:


names = ["Grade_A", "Grade_B", "Grade_C","Grade_D","Grade_F"]
cm = confusion_matrix(test_Y, predicted_Y1,labels=["A", "B", "C","D","F"])
plt.figure()
plot_confusion_matrix(cm, classes=names, title='Confusion matrix')


# In[27]:


#Plot training accuracy and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs,acc,'bo',label = 'Training Accuracy')
plt.plot(epochs,val_acc,'b',label = 'Validation Accuracy')
plt.title('Plot of Validation and Training Accuracy')
plt.xlabel("epochs ")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[26]:


#Plot training loss and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs  = range(1,len(loss)+1)
plt.plot(epochs,loss,'bo',label = 'Training Loss')
plt.plot(epochs,val_loss,'b',label = 'Validation Loss')
plt.title('Plot of Validation and Training Loss')
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[28]:


#droping g2 to study the impact of g2 on student performance 
studentData_without_G2 =  studentData_without_G3.drop(['G2'], axis=1)


# In[29]:


Y_G2 = studentData_without_G2.filter(["Grade"],axis=1)
X_G2 = studentData_without_G2.drop(['Grade'],axis=1)
xTrain_G2, xTest_G2, yTrain_G2, yTest_G2 = train_test_split(X_G2, Y_G2, test_size=0.15)


# In[30]:


#one hot encoding
x_train_G2 = pd.get_dummies(xTrain_G2)
x_test_G2 = pd.get_dummies(xTest_G2)
y_train_G2  = pd.get_dummies(yTrain_G2)
y_test_G2  = pd.get_dummies(yTest_G2)


# In[31]:


#1 input layer 2 hidden layer  
model_G2 = Sequential()
model_G2.add(Dense(64, activation='relu',  kernel_regularizer=regularizers.l2(0.001),input_shape = (57,)))
model_G2.add(Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model_G2.add(Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model_G2.add(Dense(5,  kernel_regularizer=regularizers.l2(0.001),activation='softmax'))
model_G2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_G2  = model_G2.fit(x_train_G2,y_train_G2, epochs = 200, batch_size = 5, validation_data = (x_test_G2,y_test_G2))
model_G2.save('UCI_model_B.h5')


# In[32]:


model_2 = load_model('UCI_model_B.h5')
model_2.summary()
results = model_2.evaluate(x_test_G2,y_test_G2)
print("Accuracy of the Model %.2f%%" % ( results[1]*100))


# In[33]:


#Predict Y values
y_predicted_G2 = model_2.predict(x_test_G2)

#Convert y_test dataframe to letter grades
test_Y_G2 = []
for i in y_test_G2.values:
    if i[4].round() ==1:
        test_Y_G2.append('F')
    elif i[3].round() ==1:
        test_Y_G2.append('D')
    elif i[2].round() == 1:
        test_Y_G2.append('C')
    elif i[1].round() ==1:
        test_Y_G2.append('B')
    elif i[0].round() ==1:
        test_Y_G2.append('A')
 
#convert y_predicted dataframe to letter grades
predicted_Y_G2 = []
for i in y_predicted_G2:
    if i[4].round() ==1:
        predicted_Y_G2.append('F')
    elif i[3].round() ==1:
        predicted_Y_G2.append('D')
    elif i[2].round() == 1:
        predicted_Y_G2.append('C')
    elif i[1].round() ==1:
        predicted_Y_G2.append('B')
    else:
        predicted_Y_G2.append('A')


# In[34]:


#Generate Confusion Matrix and plot the same
names = ["Grade_A", "Grade_B", "Grade_C","Grade_D","Grade_F"]
cm_G2 = confusion_matrix(test_Y_G2, predicted_Y_G2,labels=["A", "B", "C","D","F"])
plt.figure()
plot_confusion_matrix(cm_G2, classes=names, title='Confusion matrix')


# In[35]:


#Plot training accuracy and validation accuracy
acc_G2 = history_G2.history['accuracy']
val_acc_G2 = history_G2.history['val_accuracy']
plt.plot(epochs,acc_G2,'bo',label = 'Training Accuracy')
plt.plot(epochs,val_acc_G2,'b',label = 'Validation Accuracy')
plt.title('Plot of Validation and Training Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# In[36]:


#Plot training loss and validation loss
loss_G2 = history_G2.history['loss']
val_loss_G2 = history_G2.history['val_loss']
epochs  = range(1,len(loss)+1)
plt.plot(epochs,loss_G2,'bo',label = 'Training Loss')
plt.plot(epochs,val_loss_G2,'b',label = 'Validation Loss')
plt.title('Plot of Validation and Training Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[37]:


#droping g1
studentData_without_G1 =  studentData_without_G2.drop(['G1'], axis=1)


# In[39]:


Y_G1 = studentData_without_G1.filter(["Grade"],axis=1)
X_G1 = studentData_without_G1.drop(['Grade'],axis=1)

#split the dataset into train and test
xTrain_G1, xTest_G1, yTrain_G1, yTest_G1 = train_test_split(X_G1, Y_G1, test_size=0.15)


# In[40]:


x_train_G1 = pd.get_dummies(xTrain_G1)
x_test_G1 = pd.get_dummies(xTest_G1)
y_train_G1  = pd.get_dummies(yTrain_G1)
y_test_G1 = pd.get_dummies(yTest_G1)


# In[41]:


model_G1 = Sequential()
model_G1.add(Dense(64, activation='relu',  kernel_regularizer=regularizers.l2(0.001),input_shape = (56,)))
model_G1.add(Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model_G1.add(Dense(32, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model_G1.add(Dense(5,  kernel_regularizer=regularizers.l2(0.001),activation='softmax'))
model_G1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_G1  = model_G1.fit(x_train_G1,y_train_G1, epochs = 200, batch_size = 5, validation_data = (x_test_G1,y_test_G1))
model_G1.save('UCI_model_C.h5')


# In[42]:


model_3 = load_model('UCI_model_C.h5')
model_3.summary()
results = model_3.evaluate(x_test_G1,y_test_G1)
print("Accuracy of the Model %.2f%%" % ( results[1]*100))


# In[43]:


#Predict Y values
y_predicted_G1 = model_3.predict(x_test_G1)

#Convert y_test dataframe to letter grades
test_Y_G1 = []
for i in y_test_G1.values:
    if i[4].round() ==1:
        test_Y_G1.append('F')
    elif i[3].round() ==1:
        test_Y_G1.append('D')
    elif i[2].round() == 1:
        test_Y_G1.append('C')
    elif i[1].round() ==1:
        test_Y_G1.append('B')
    elif i[0].round() ==1:
        test_Y_G1.append('A')
 
#convert y_predicted dataframe to letter grades
predicted_Y_G1 = []
for i in y_predicted_G1:
    if i[4].round() ==1:
        predicted_Y_G1.append('F')
    elif i[3].round() ==1:
        predicted_Y_G1.append('D')
    elif i[2].round() == 1:
        predicted_Y_G1.append('C')
    elif i[1].round() ==1:
        predicted_Y_G1.append('B')
    else:
        predicted_Y_G1.append('A')


# In[44]:


#Generate Confusion Matrix and plot the same
names = ["Grade_A", "Grade_B", "Grade_C","Grade_D","Grade_F"]
cm_G1 = confusion_matrix(test_Y_G1, predicted_Y_G1,labels=["A", "B", "C","D","F"])
plt.figure()
plot_confusion_matrix(cm_G1, classes=names, title='Confusion matrix')


# In[ ]:




