#import autosklearn.classification
import numpy as np 
import joblib
from sklearn.linear_model import LinearRegression 
#Acquisition des données 
pageSpeeds=np.random.normal(3.0,1.0,10)
purchaseAmount=100-(pageSpeeds+np.random.normal(0,0.1,10))*3
#Entraînement du modèle 
model=LinearRegression()
model.fit(pageSpeeds.reshape(-1,1),purchaseAmount)

#Sérialisation du modèle pour utilisation future
joblib.dump(model,'./model.joblib')

#Prédiction  => c'est le boullot du predictor.py
"""
purchasePredict=model.predict([[7.0]])

print("Prediction={:.4f}".format(purchasePredict[0]))
"""
