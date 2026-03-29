import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv('emails.csv')
X = df.drop(['Email No.', 'Prediction'], axis=1)
y=df["Prediction"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
xgb=XGBClassifier(random_state=42)
gbm_param_grid = {
'learning_rate': np.arange(0.05, 1.05, .05),
'n_estimators' : np.arange(50,200,50),
'max_depth': [3, 4, 5, 6, 7, 8, 9],
'subsample' : np.arange(0.05, 1.05, .05),
'colsample_bytree':np.arange(0.05, 1.05, .05),
'reg_alpha': [0, 0.1, 0.5, 1],
'reg_lambda': [1, 1.5, 2, 5]
}
model = RandomizedSearchCV(estimator=xgb,
param_distributions=gbm_param_grid,
n_iter=200, scoring='f1',cv=4, verbose=1,n_jobs=-1,random_state=42)
model.fit(X_train,y_train)
# Meilleurs hyperparamètres
print("Best params:", model.best_params_)
# Meilleur score F1
print("Best score:", model.best_score_)
best_model = model.best_estimator_
y_pred=best_model.predict(X_test)
print(classification_report(y_test, y_pred))
# Sauvegarder X_test sans labels
X_test.to_csv('X_test.csv', index=False)
print(f"X_test sauvegardé ✅")
# Sauvegarder y_test séparément
y_test.to_csv('y_test.csv', index=False)
print(f"y_test sauvegardé ✅")
# Sauvegarder le modèle final
joblib.dump(best_model, 'spam_model1.pkl')
joblib.dump(X.columns.tolist(), 'feature_names1.pkl')
print("Modèle sauvegardé ✅")
