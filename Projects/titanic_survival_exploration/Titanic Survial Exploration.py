import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rnd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df,test_df]
train_df.shape
test_df.shape
combine[0].shape
combine[1].shape

train_df['Age'].plot()
train_df['Age'].plot(kind='line')
train_df['Age'].plot(kind='scatter')
train_df['Age'].plot(kind='box')

train_df.head()
train_df.tail()
train_df.describe()
train_df.describe(include=['O'])

train_df[['Pclass','Survived']].groupby(['Pclass'],
        as_index=False).mean().sort_values(by='Survived',ascending=True)
train_df[['Sex','Survived']].groupby(['Sex'],
        as_index=False).mean().sort_values(by='Sex',ascending=False)
train_df[['SibSp','Survived']].groupby(['SibSp'],
        as_index=False).mean().sort_values(by='SibSp',ascending=False)
train_df[['Parch','Survived']].groupby(['Parch'],
        as_index=False).mean().sort_values(by='Parch',ascending=False)

g = sns.FacetGrid(train_df,col='Survived')
g.map(plt.hist,'Age',bins=20)
plt.show()
g = sns.FacetGrid(train_df,col='Survived',row='Pclass',size=2.2,aspect=1.6)
g.map(plt.hist,'Age',bins=20)
plt.show()
g = sns.FacetGrid(train_df,row='Embarked',size=2.2,aspect=1.6)
g.map(sns.pointplot,'Pclass','Survived','Sex',palette='deep')
g.add_legend()
g = sns.FacetGrid(train_df,row='Embarked',size=2.2,aspect=1.6)
g.map(sns.barplot,'Sex','Fare',alpha=0.5,ci=None)
g.add_legend()

print('before',train_df.shape,test_df.shape,combine[0].shape,combine[1].shape)
train_df = train_df.drop(['Ticket','Cabin'],axis=1)
test_df = test_df.drop(['Ticket','Cabin'],axis=1)
combine = [train_df,test_df]
print('after',train_df.shape,test_df.shape,combine[0].shape,combine[1].shape)

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.',expand=False)
print(train_df.shape,test_df.shape,combine[0].shape,combine[1].shape)
pd.crosstab(train_df['Title'],train_df['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
           'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')
train_df[['Title','Survived']].groupby(['Title'],
        as_index=False).mean().sort_values(by='Survived',ascending=False)

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train_df.head()

print('before',train_df.shape,test_df.shape,combine[0].shape,combine[1].shape)
train_df = train_df.drop(['Name','PassengerId'],axis=1)
test_df = test_df.drop(['Name'],axis=1)
combine = [train_df,test_df]
print('after',train_df.shape,test_df.shape,combine[0].shape,combine[1].shape)

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female':1,'male':0}).astype(int)
train_df.head()

g = sns.FacetGrid(train_df,row='Pclass',col='Sex',size=2.2,aspect=1.6)
g.map(plt.hist,'Age',alpha=0.5,bins=20)
g.add_legend()

guess_ages = np.zeros((2,3))
guess_ages

for dataset in combine:
    for i in range(0,2):
        for j in range(0,3):
            guess_df = dataset[(dataset['Sex']==i) & 
                               (dataset['Pclass']==j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            guess_ages[i,j] = int( age_guess/0.5 + 0.5) * 0.5
guess_ages           
for dataset in combine:
   for i in range(0,2):
       for j in range(0,3):
           dataset.loc[(dataset.Age.isnull()) & (dataset.Sex ==i) & (dataset.Pclass ==j+1),\
                       'Age'] = guess_ages[i,j]
dataset['Age'] = dataset['Age'].astype(int)
train_df.head(10)

train_df['AgeBand'] = pd.cut(train_df['Age'],5)
train_df[['AgeBand','Survived']].groupby(['AgeBand'],
        as_index=False).mean().sort_values(by='Survived',ascending=False)

for dataset in combine:
    dataset.loc[dataset['Age'] <= 16 , 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32),'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48),'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64),'Age'] = 3
    dataset.loc[dataset['Age'] > 64,'Age'] 
train_df.head(10)

train_df = train_df.drop(['AgeBand'],axis=1)
combine = [train_df,test_df]

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
train_df[['FamilySize','Survived']].groupby(['FamilySize'],
        as_index=False).mean().sort_values(by='Survived',ascending=False) 

for dataset in combine: 
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize']==1,'IsAlone'] = 1
train_df[['IsAlone','Survived']].groupby(['IsAlone'],
        as_index=False).mean().sort_values(by='Survived',ascending=False)    

train_df = train_df.drop(['Parch','SibSp','FamilySize'],axis=1)
test_df = test_df.drop(['Parch','SibSp','FamilySize'],axis=1)
combine = [train_df,test_df]

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
train_df.loc[:,['Age*Class','Age','Pclass']].head(10)   

freq_port = train_df.Embarked.dropna().mode()[0]
freq_port

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
train_df[['Embarked','Survived']].groupby(['Embarked'],
        as_index=False).mean().sort_values(by='Survived',ascending=False)  

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
train_df.head()

test_df['Fare'].fillna(test_df['Fare'].dropna().median(),inplace=True)
test_df.head()

train_df['FareBand'] = pd.cut(train_df['Fare'],4)
train_df[['FareBand','Survived']].groupby(['FareBand'],
        as_index=False).mean().sort_values(by='FareBand',ascending=False)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454),'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31),'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 31) ,'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train_df = train_df.drop(['FareBand'],axis=1)
combine = [train_df,test_df]
train_df.shape,test_df.shape


X_train = train_df.drop('Survived',axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId',axis=1).copy()
X_train.shape,Y_train.shape,X_test.shape

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train,Y_train)*100,2)
acc_log

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['correlations'] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='correlations',ascending=False)

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train,Y_train)*100,2)
acc_svc

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train,Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train,Y_train)*100,2)
acc_knn

from sklearn.naive_bayes import GaussianNB
gaussian = GaussianNB()
gaussian.fit(X_train,Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train,Y_train)*100,2)
acc_gaussian

from sklearn.linear_model import Perceptron
perceptron = Perceptron()
perceptron.fit(X_train,Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train,Y_train)*100,2)
acc_perceptron

from sklearn.svm import LinearSVC
linear_svc = LinearSVC()
linear_svc.fit(X_train,Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train,Y_train)*100,2)
acc_linear_svc

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(X_train,Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train,Y_train)*100,2)
acc_sgd

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train,Y_train)*100,2)
acc_decision_tree

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train,Y_train)*100,2)
acc_random_forest

models = pd.DataFrame({
        'Model':['Support Vector Machines', 'KNN', 'Logistic Regression',
                 'Random Forest', 'Naive Bayes', 'Perceptron',
                 'Stochastic Gradient Decent', 'Linear SVC',
                 'Decision Tree'],
         'Score':[acc_svc, acc_knn, acc_log,
                  acc_random_forest, acc_gaussian, acc_perceptron,
                  acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score',ascending=False)
    
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = decision_tree, X = X_train, y = Y_train, cv = 10)
meanda = accuracies.mean()
deviationda = accuracies.std()

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
decision_tree = DecisionTreeClassifier()
from sklearn.metrics import f1_score, make_scorer

parameters = {'max_depth':[2,4,6,8,10],
              'min_samples_leaf':[2,4,6,8,10],
              'min_samples_split':[2,4,6,8,10]}

scorer = make_scorer(f1_score)

grid_obj = GridSearchCV(decision_tree,parameters,scoring=scorer)

grid_fit = grid_obj.fit(X_train,Y_train)

best_fit = grid_fit.best_estimator_

best_fit.fit(X_train,Y_train)

best_train_predictions = best_fit.predict(X_train)
best_test_predictions = best_fit.predict(X_test)

from sklearn.metrics import confusion_matrix
best_cn_train = confusion_matrix(Y_train,best_train_predictions)

best_f1score_train = f1_score(Y_train,best_train_predictions)

plot_model(X, y, best_clf)



    
























           

    


    
    

    