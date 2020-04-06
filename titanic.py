import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
test_pasengerID = test_df["PassengerId"]


def detect_outliers(df,features):
    outlier = []
    for i in features :

        q1 = np.percentile(df[i],25)
        q3 = np.percentile(df[i],75)
        IQR = q3-q1
        outlier_step = IQR*1.5
        outlier_list_col = df[(df[i]<q1 - outlier_step) | (df[i]>q3+outlier_step)].index
        outlier.extend(outlier_list_col)

    outlier = Counter(outlier)
    multiple_outliers = list(i for i,v in outlier.items() if v>2)
    return multiple_outliers



train_df = train_df.drop(detect_outliers(train_df,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)

train_df = pd.concat([train_df,test_df],axis=0).reset_index(drop = True)


train_df["Embarked"] = train_df["Embarked"].fillna("C")
train_df["Fare"] = train_df["Fare"].fillna(np.mean(train_df[train_df["Pclass"]==3]["Fare"]))


#fill the missing data


index_nan_age = list(train_df["Age"][train_df["Age"].isnull()].index)
for i in index_nan_age:
    age_pred = train_df["Age"][((train_df["SibSp"] == train_df.iloc[i]["SibSp"]) &(train_df["Parch"] == train_df.iloc[i]["Parch"])& (train_df["Pclass"] == train_df.iloc[i]["Pclass"]))].median()
    age_med = train_df["Age"].median()
    if not np.isnan(age_pred):
        train_df["Age"].iloc[i] = age_pred
    else:
        train_df["Age"].iloc[i] = age_med


name = train_df["Name"]
train_df["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in name]
# convert to categorical
train_df["Title"] = train_df["Title"].replace(["Lady","the Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"],"other")
train_df["Title"] = [0 if i == "Master" else 1 if i == "Miss" or i == "Ms" or i == "Mlle" or i == "Mrs" else 2 if i == "Mr" else 3 for i in train_df["Title"]]
train_df["Title"].head(20)

train_df.drop(labels = ["Name"], axis = 1, inplace = True)
train_df = pd.get_dummies(train_df,columns=["Title"])
train_df["Fsize"] = train_df["SibSp"] + train_df["Parch"] + 1
train_df["family_size"] = [1 if i < 5 else 0 for i in train_df["Fsize"]]
train_df = pd.get_dummies(train_df, columns= ["family_size"])
train_df = pd.get_dummies(train_df, columns=["Embarked"])

tickets = []
for i in list(train_df.Ticket):
    if not i.isdigit():
        tickets.append(i.replace(".","").replace("/","").strip().split(" ")[0])
    else:
        tickets.append("x")
train_df["Ticket"] = tickets
train_df = pd.get_dummies(train_df, columns= ["Ticket"], prefix = "T")

train_df["Pclass"] = train_df["Pclass"].astype("category")
train_df = pd.get_dummies(train_df, columns= ["Pclass"])

train_df["Sex"] = train_df["Sex"].astype("category")
train_df = pd.get_dummies(train_df, columns=["Sex"])

train_df.drop(labels = ["PassengerId", "Cabin"], axis = 1, inplace = True)


test = train_df[881:]
test.drop(labels = ["Survived"],axis = 1, inplace = True)
train = train_df[:881]
X_train = train.drop(labels = "Survived", axis = 1)
y_train = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.33, random_state = 42)




logreg = LogisticRegression()
logreg.fit(X_train, y_train)


acc_log_train = round(logreg.score(X_train, y_train)*100,2) 
acc_log_test = round(logreg.score(X_test,y_test)*100,2)
print("Training Accuracy: % {}".format(acc_log_train))
print("Testing Accuracy: % {}".format(acc_log_test))
















