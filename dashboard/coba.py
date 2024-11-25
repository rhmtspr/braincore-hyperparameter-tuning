import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score

from mealpy import FloatVar, StringVar, Problem


from mealpy.bio_based.SMA import OriginalSMA

class SGDOptimizedProblem(Problem):
    def __init__(self, bounds=None, minmax="max", data=None, classifier_name='SGD', **kwargs):
        self.data = data
        self.classifier_name = classifier_name
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        penalty, alpha, l1_ratio, loss = x_decoded["penalty"], x_decoded["alpha"], x_decoded["l1_ratio"], x_decoded["loss"]
        sgd = SGDClassifier(penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, loss=loss, random_state=55)
        sgd.fit(self.data["X_train"], self.data["y_train"])
        y_predict = sgd.predict(self.data["X_test"])
        return accuracy_score(self.data["y_test"], y_predict)

def main():
    df = pd.read_csv('alzheimers_disease_data.csv')
    df = df.drop(['PatientID', 'DoctorInCharge'], axis=1)
    X = df.drop(columns='Diagnosis').values
    y = df['Diagnosis'].values

    # Create training and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

    data = {
        "X_train": X_train_std,
        "X_test": X_test_std,
        "y_train": y_train,
        "y_test": y_test
    }

    my_bounds = [
        FloatVar(lb=0.00000001, ub=10, name="alpha"),
        FloatVar(lb=0, ub=1, name="l1_ratio"),
        StringVar(valid_sets=('l1', 'l2', 'elasticnet'), name="penalty"),
        StringVar(valid_sets=('hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'), name="loss")
    ]
    
    problem = SGDOptimizedProblem(bounds=my_bounds, minmax="max", data=data)
    model = OriginalSMA(epoch=10, pop_size=100)
    model.solve(problem)
    return model.history.epoch, model.history.list_current_best_fit, model.history.list_global_best_fit


if __name__ == "__main__":
    epoch, current_best, global_best = main()
