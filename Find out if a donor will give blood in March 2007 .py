import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

url = "https://code.datasciencedojo.com/datasciencedojo/datasets/raw/master/Blood%20Transfusion%20Service%20Center/transfusion.data.csv"
data = pd.read_csv(url)

X = data.drop(columns=["whether he/she donated blood in March 2007"])
y = data["whether he/she donated blood in March 2007"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg_model = LogisticRegression(max_iter=1000, random_state=42)
logreg_model.fit(X_train, y_train)
logreg_y_pred = logreg_model.predict(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_y_pred = gb_model.predict(X_test)

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)

xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_y_pred = xgb_model.predict(X_test)

lgb_model = LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)
lgb_y_pred = lgb_model.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

print("Evaluation on Logistic Regression:")
evaluate_model(y_test, logreg_y_pred, "Logistic Regression")

print("\nEvaluation on Random Forest:")
evaluate_model(y_test, rf_y_pred, "Random Forest")

print("\nEvaluation on Gradient Boosting:")
evaluate_model(y_test, gb_y_pred, "Gradient Boosting")

print("\nEvaluation on Support Vector Machine (SVM):")
evaluate_model(y_test, svm_y_pred, "SVM")

print("\nEvaluation on XGBoost:")
evaluate_model(y_test, xgb_y_pred, "XGBoost")

print("\nEvaluation on LightGBM:")
evaluate_model(y_test, lgb_y_pred, "LightGBM")



After tuning : 

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

url = "https://code.datasciencedojo.com/datasciencedojo/datasets/raw/master/Blood%20Transfusion%20Service%20Center/transfusion.data.csv"
data = pd.read_csv(url)

X = data.drop(columns=["whether he/she donated blood in March 2007"])
y = data["whether he/she donated blood in March 2007"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)

rf_param_grid = {
    'n_estimators': [100, 200, 300],       
    'max_depth': [None, 5, 10, 15],      
    'min_samples_split': [2, 5, 10],       
    'min_samples_leaf': [1, 2, 4],          
    'bootstrap': [True, False]              
}

rf_random = RandomizedSearchCV(estimator=rf_model, param_distributions=rf_param_grid, n_iter=10, cv=3, random_state=42)
rf_random.fit(X_train, y_train)
rf_y_pred = rf_random.predict(X_test)

gb_model = GradientBoostingClassifier(random_state=42)

gb_param_grid = {
    'n_estimators': [100, 200, 300],       
    'learning_rate': [0.01, 0.1, 0.2, 0.3], 
    'max_depth': [3, 5, 7],                 
    'min_samples_split': [2, 5, 10],        
    'min_samples_leaf': [1, 2, 4],          
}

gb_random = RandomizedSearchCV(estimator=gb_model, param_distributions=gb_param_grid, n_iter=10, cv=3, random_state=42)
gb_random.fit(X_train, y_train)
gb_y_pred = gb_random.predict(X_test)

def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"{model_name} - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

print("Evaluation on Random Forest with Hyperparameter Tuning:")
evaluate_model(y_test, rf_y_pred, "Random Forest")

print("\nEvaluation on Gradient Boosting with Hyperparameter Tuning:")
evaluate_model(y_test, gb_y_pred, "Gradient Boosting")
