from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

def tune_hyperparameters(X, y):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=2)
    grid_search.fit(X, y)
    return grid_search.best_params_, grid_search.best_score_

if __name__ == "__main__":
    data = pd.read_csv("data/processed_data.csv")
    X = data.drop('quality', axis=1)
    y = data['quality']

    best_params, best_score = tune_hyperparameters(X, y)
    print(f"Best Parameters: {best_params}")
    print(f"Best CV Score: {best_score}")

