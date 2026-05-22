# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect File Encoding: Use chardet to determine the dataset's encoding.
2. Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3. Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4. Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5. Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6. Train SVM Model: Fit an SVC model on the training data.
7. Predict Labels: Predict test labels using the trained SVM model.
8. Evaluate Model: Calculate and display accuracy with metrics.accuracy_score. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: BALAJI S
RegisterNumber:  212225220015
*/
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
df=pd.read_csv(r"C:\Users\acer\Downloads\spam.csv",encoding='latin-1')
df.head()
```

<img width="907" height="264" alt="image" src="https://github.com/user-attachments/assets/41568fc5-4a33-431d-b5f1-0fb07f3867bf" />

```
df=df[['v1','v2']]
df
```

<img width="612" height="562" alt="image" src="https://github.com/user-attachments/assets/3c9a56ba-c3e3-439e-a04b-c7b4bdfeb055" />

```
df.columns=['label','message']
df
```

<img width="597" height="553" alt="image" src="https://github.com/user-attachments/assets/159bb6ba-ebcf-450a-b81a-344f1a249056" />

```
df['label']=df['label'].map({'ham':0,'spam':1})
print("\nDataSet Shape:",df.shape)
print(df.head())
x=df['message']
y=df['label']
vectorizer=TfidfVectorizer(stop_words='english')
x=vectorizer.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=SVC(kernel='linear')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
print("Classification Report:")
print(classification_report(y_test,y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
sample_mail=["Congratulations ! You won a free Iphone. click now."]
sample_data=vectorizer.transform(sample_mail)
prediction=model.predict(sample_data)
print("Custom Mail Prediction:")
if(prediction[0]==1):
    print("Spam mail")
else:
    print("Not spam mail")
```

<img width="777" height="658" alt="image" src="https://github.com/user-attachments/assets/a09404a8-e047-446c-bfee-c0a977545a97" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
