"""
Simple training script:
- loads iris dataset from sklearn
- trains a LogisticRegression
- saves model to model.pkl
"""

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os
import json

def main():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Save model
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "model.pkl")
    joblib.dump(model, model_path)

    # Save a tiny metrics file
    acc = model.score(X_test, y_test)
    metrics = {"accuracy": float(acc)}
    with open(os.path.join("artifacts", "metrics.json"), "w") as f:
        json.dump(metrics, f)

    print(f"Saved model to {model_path}")
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()

# Understanding the if __name__ == "__main__": Pattern
# This is a Python idiom that controls when your code runs. 
# It's one of the most important patterns in Python, especially for scripts and modules.

# What It Does --The condition if __name__ == "__main__": 
# checks whether the current file is being run directly (as a script) 
# versus being imported as a module into another file. 
# When you run train.py directly from the command line or terminal, 
# Python sets the special variable __name__ to the string "__main__". 
# However, if another file imports train.py, 
# the __name__ variable is set to the module's name (in this case, "train").