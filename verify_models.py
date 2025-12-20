from models import ModelHandler
import os

def verify():
    print("Initializing ModelHandler...")
    handler = ModelHandler()
    
    train_path = "Datasets/train.csv"
    test_path = "Datasets/test.csv"
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    print("Loading data...")
    msg = handler.load_and_preprocess(train_path)
    print(msg)
    
    print("\nTesting KNN Training...")
    acc, report, cm = handler.train_knn(k=3)
    print(f"KNN Accuracy: {acc}")
    
    print("\nTesting Logistic Regression Training...")
    acc, report, cm = handler.train_logistic_regression()
    print(f"LR Accuracy: {acc}")
    
    print("\nTesting Random Forest Training...")
    acc, report, cm = handler.train_random_forest(n_estimators=10)
    print(f"RF Accuracy: {acc}")
    
    print("\nTesting Evaluation on Test File...")
    acc, report, cm = handler.evaluate_on_test_file(test_path)
    print(f"Test File Accuracy: {acc}")
    
    print("\nVerification Complete!")

if __name__ == "__main__":
    verify()
