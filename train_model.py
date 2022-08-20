import json
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

with open('params.yaml', 'rb') as f:
    params = yaml.safe_load(f)

dataset = 'winequality-red.csv'

df = pd.read_csv(dataset, header = 0, sep = ';')

y = df.pop('quality')
X = df

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size = 0.25,
    stratify = y
)

end = OneHotEncoder(sparse = False, handle_unknown = 'ignore')

y_train = end.fit_transform(y_train.values.reshape(-1, 1))
y_test = end.transform(y_test.values.reshape(-1, 1))

rf = RandomForestClassifier(n_estimators = params['n_estimators'])

rf.fit(X_train, y_train)

pred = rf.predict(X_test)

f1 = f1_score(y_test, pred, average = 'micro')

with open('f1.json', 'w') as f:
    json.dump({'f1': f1}, f)

print(f'F1 score: {f1:.4f}')


