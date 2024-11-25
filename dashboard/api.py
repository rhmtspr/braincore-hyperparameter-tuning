import io
import base64
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from mealpy import FloatVar, IntegerVar, Problem
from mealpy.bio_based.SMA import OriginalSMA

import xgboost as xgb
from xgboost import XGBClassifier

np.random.seed(42)
random.seed(42)

class XGBoostOptimizedProblem(Problem):
    def __init__(self, bounds=None, minmax="max", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        n_estimators = x_decoded["n_estimators"]
        max_depth = x_decoded["max_depth"]
        learning_rate = x_decoded["learning_rate"]
        min_child_weight = x_decoded["min_child_weight"]
        subsample = x_decoded["subsample"]
        colsample_bytree = x_decoded["colsample_bytree"]
        
        xgb_classifier = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=55
        )
        
        xgb_classifier.fit(self.data["X_train"], self.data["y_train"])
        y_predict = xgb_classifier.predict(self.data["X_test"])
        return accuracy_score(self.data["y_test"], y_predict)

class OptimizationRequest(BaseModel):
    target_column: str
    scaling_method: Optional[str] = "None"
    epochs: int = 50
    population_size: int = 100
    max_early_stop: int = 10

class OptimizationResult(BaseModel):
    best_accuracy: float
    best_parameters: Dict[str, Any]
    classification_report: Dict[str, Any]
    confusion_matrix: List[List[int]]
    confusion_matrix_image: Optional[str] = None

def hyperparameter_tuning(data, epoch=50, pop_size=100, max_early_stop=10):
    my_bounds = [
        IntegerVar(lb=10, ub=300, name="n_estimators"),
        IntegerVar(lb=1, ub=15, name="max_depth"),
        FloatVar(lb=0.01, ub=1, name="learning_rate"),
        IntegerVar(lb=1, ub=10, name="min_child_weight"),
        FloatVar(lb=0.5, ub=1, name="subsample"),
        FloatVar(lb=0.5, ub=1, name="colsample_bytree")
    ]
    
    problem = XGBoostOptimizedProblem(
        bounds=my_bounds, 
        minmax="max", 
        data=data,         
        verbose=True, 
        save_population=True
    )

    model = OriginalSMA(epoch=epoch, pop_size=pop_size)

    term_dict = {
        "max_fe": 10000,
        "max_early_stop": max_early_stop
    }

    model.solve(problem, termination=term_dict, seed=10)
    
    print(f"Best accuracy: {model.g_best.target.fitness}")
    print(f"Best parameters: {model.problem.decode_solution(model.g_best.solution)}")
    
    return model

app = FastAPI()

@app.post("/optimize", response_model=OptimizationResult)
async def optimize_hyperparameters(request: OptimizationRequest):
    try:
        data_path = 'data/alzheimers_disease_data.csv'
        Data = pd.read_csv(data_path)
        
        columns_to_drop = ['DoctorInCharge', 'PatientId']
        Data = Data.drop(columns=columns_to_drop, errors='ignore')

        feature_columns = [col for col in Data.columns if col != 'Diagnosis']
        X = Data[feature_columns]
        y = Data['Diagnosis']
        
        # Scaling
        if request.scaling_method != "None":
            scalers = {
                "Standard Scaling": StandardScaler(),
                "Min-Max Scaling": MinMaxScaler(),
                "Robust Scaling": RobustScaler()
            }
            scaler = scalers[request.scaling_method]
            X = scaler.fit_transform(X)

        # Train-Test Split
        X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        data = {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test
        }

        # Hyperparameter tuning
        model = hyperparameter_tuning(
            data=data, 
            epoch=request.epochs, 
            pop_size=request.population_size, 
            max_early_stop=request.max_early_stop
        )

        param = model.problem.decode_solution(model.g_best.solution)
                
        xgb_classifier = XGBClassifier(
            n_estimators=int(param['n_estimators']),
            max_depth=int(param['max_depth']),
            learning_rate=float(param['learning_rate']),
            min_child_weight=int(param['min_child_weight']),
            subsample=float(param['subsample']),
            colsample_bytree=float(param['colsample_bytree']),
            random_state=48
        )
                
        xgb_classifier.fit(data["X_train"], data["y_train"])
        y_pred = xgb_classifier.predict(data["X_test"])

        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=["Class 0", "Class 1"], 
                    yticklabels=["Class 0", "Class 1"])
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        
        # Convert plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        conf_matrix_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Convert classification report to primitive types
        converted_report = {}
        for key, value in report.items():
            if isinstance(value, dict):
                converted_report[key] = {k: float(v) for k, v in value.items()}
            else:
                converted_report[key] = float(value)

        return {
            "best_accuracy": float(accuracy_score(y_test, y_pred)),
            "best_parameters": {k: float(v) if isinstance(v, (np.float64, np.float32)) else int(v) for k, v in param.items()},
            "classification_report": converted_report,
            "confusion_matrix": conf_matrix.tolist(),
            "confusion_matrix_image": conf_matrix_image
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "XGBoost Hyperparameter Tuning API with SMA is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)