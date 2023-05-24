from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('cardio_train.csv', sep=";")  # Replace 'path_to_dataset.csv' with the actual path to your dataset
dataset.drop("id",axis=1,inplace=True)
dataset.dropna(inplace=True)

dataset.drop_duplicates(inplace=True)

dataset.describe()

X = dataset.drop('cardio', axis=1)
y = dataset['cardio']

train, test, target, target_test = train_test_split(X, y, test_size=0.2, random_state=0)

random_forest = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': [100, 300]}, cv=5).fit(train, target)
random_forest.fit(train, target)
y = random_forest.predict(test)
print(y)
acc_random_forest = round(random_forest.score(train, target) * 100, 2)
print(acc_random_forest,random_forest.best_params_)
acc_test_random_forest = round(random_forest.score(test, target_test) * 100, 2)
print(acc_test_random_forest)

joblib.dump(random_forest, 'clasificator.pkl')