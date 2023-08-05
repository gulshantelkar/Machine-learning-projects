import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


data_url = 'https://code.datasciencedojo.com/datasciencedojo/datasets/raw/master/Default%20of%20Credit%20Card%20Clients/default%20of%20credit%20card%20clients.csv'
df = pd.read_csv(data_url)


df.rename(columns={'default payment next month': 'Y'}, inplace=True)

X = df.drop('Y', axis=1) 


label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Y'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


logistic_model = LogisticRegression()
random_forest_model = RandomForestClassifier()
gradient_boosting_model = GradientBoostingClassifier()
svm_model = SVC(probability=True)


logistic_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)
gradient_boosting_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probabilities)

    return accuracy, precision, recall, f1, roc_auc

print("Logistic Regression:")
acc, prec, rec, f1, roc_auc = evaluate_model(logistic_model, X_test, y_test)
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)

print("\nRandom Forest:")
acc, prec, rec, f1, roc_auc = evaluate_model(random_forest_model, X_test, y_test)
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)

print("\nGradient Boosting:")
acc, prec, rec, f1, roc_auc = evaluate_model(gradient_boosting_model, X_test, y_test)
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)

print("\nSupport Vector Machine:")
acc, prec, rec, f1, roc_auc = evaluate_model(svm_model, X_test, y_test)
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)





after Hypertuning :
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

data_url = 'https://code.datasciencedojo.com/datasciencedojo/datasets/raw/master/Default%20of%20Credit%20Card%20Clients/default%20of%20credit%20card%20clients.csv'
df = pd.read_csv(data_url)

df.rename(columns={'default payment next month': 'Y'}, inplace=True)

X = df.drop('Y', axis=1)  

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Y'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

logistic_params = {
    'C': [0.01, 0.1, 1],
    'penalty': ['l2']
}
logistic_model = GridSearchCV(LogisticRegression(max_iter=1000), param_grid=logistic_params, cv=5)
logistic_model.fit(X_train, y_train)

random_forest_params = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [5, 10]
}
random_forest_model = GridSearchCV(RandomForestClassifier(), param_grid=random_forest_params, cv=5)
random_forest_model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probabilities)

    return accuracy, precision, recall, f1, roc_auc

print("Logistic Regression:")
acc, prec, rec, f1, roc_auc = evaluate_model(logistic_model, X_test, y_test)
print("Best Parameters:", logistic_model.best_params_)
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)

print("\nRandom Forest:")
acc, prec, rec, f1, roc_auc = evaluate_model(random_forest_model, X_test, y_test)
print("Best Parameters:", random_forest_model.best_params_)
print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("ROC-AUC:", roc_auc)
