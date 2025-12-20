import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing import preprocess

class ModelHandler:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.scaler = StandardScaler()
        self.encoders = None
        self.feature_cols = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_preprocess(self, train_path, test_path=None):
        # Load Train Data
        df_train = pd.read_csv(train_path)
        df_processed, self.encoders = preprocess(df_train, fit=True)
        
        self.feature_cols = [
            col for col in df_processed.columns 
            if col not in ['price', 'price_encoded'] and df_processed[col].dtype != 'object'
        ]
        
        X = df_processed[self.feature_cols]
        y = df_processed['price_encoded']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale Data
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return "Data Loaded and Preprocessed Successfully."

    def train_knn(self, k=5):
        self.model_type = 'KNN'
        self.model = KNeighborsClassifier(n_neighbors=k)
        self.model.fit(self.X_train_scaled, self.y_train)
        return self._evaluate_internal()

    def train_logistic_regression(self, max_iter=300):
        self.model_type = 'Logistic Regression'
        self.model = LogisticRegression(max_iter=max_iter)
        self.model.fit(self.X_train_scaled, self.y_train)
        return self._evaluate_internal()

    def train_random_forest(self, n_estimators=200, max_depth=20):
        self.model_type = 'Random Forest'
        # RF doesn't strictly need scaling, but we use the scaled data for consistency in pipeline
        # or we can use unscaled. Let's use unscaled for RF as per original script, 
        # but for simplicity in this handler, using scaled won't hurt much, 
        # HOWEVER, original script used X_train (unscaled). Let's stick to that for RF.
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
        self.model.fit(self.X_train, self.y_train)
        
        # Internal eval
        pred = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, pred)
        report = classification_report(self.y_test, pred)
        cm = confusion_matrix(self.y_test, pred)
        return acc, report, cm

    def _evaluate_internal(self):
        pred = self.model.predict(self.X_test_scaled)
        acc = accuracy_score(self.y_test, pred)
        report = classification_report(self.y_test, pred)
        cm = confusion_matrix(self.y_test, pred)
        return acc, report, cm

    def evaluate_on_test_file(self, test_path):
        if not self.model:
            return "Model not trained yet."
            
        testdf = pd.read_csv(test_path)
        yt = testdf['price']
        yt_encoded = yt.map({'expensive': 1, 'non-expensive': 0})
        
        testdf_features = testdf.drop(columns=['price'])
        test_processed, _ = preprocess(testdf_features, label_encoders=self.encoders, fit=False)
        
        # Align columns
        for col in self.feature_cols:
            if col not in test_processed.columns:
                test_processed[col] = 0
        
        X_test_final = test_processed[self.feature_cols]
        
        if self.model_type == 'Random Forest':
            # RF used unscaled in training
            y_pred = self.model.predict(X_test_final)
        else:
            # Others used scaled
            X_test_final_scaled = self.scaler.transform(X_test_final)
            y_pred = self.model.predict(X_test_final_scaled)
            
        acc = accuracy_score(yt_encoded, y_pred)
        report = classification_report(yt_encoded, y_pred)
        cm = confusion_matrix(yt_encoded, y_pred)
        
        return acc, report, cm

    def get_feature_importance(self):
        if self.model_type == 'Random Forest':
            importances = self.model.feature_importances_
            feature_names = self.feature_cols
            return feature_names, importances
        return None, None

    def predict_single(self, features_dict):
        """
        Predict price for a single instance.
        features_dict: dictionary of feature names and values
        """
        if not self.model:
            return "Model not trained."
            
        # Create DataFrame from input
        input_df = pd.DataFrame([features_dict])
        
        # Ensure categorical columns exist
        categorical_columns = ['Processor_Brand', 'Processor_Series', 'Notch_Type', 'os_name', 'brand']
        for col in categorical_columns:
            if col not in input_df.columns:
                input_df[col] = "Unknown"

        # Preprocess (using existing encoders)
        processed_df, _ = preprocess(input_df, label_encoders=self.encoders, fit=False)
        
        # Align columns
        for col in self.feature_cols:
            if col not in processed_df.columns:
                processed_df[col] = 0
                
        X_final = processed_df[self.feature_cols]
        
        if self.model_type == 'Random Forest':
            prediction = self.model.predict(X_final)[0]
        else:
            X_final_scaled = self.scaler.transform(X_final)
            prediction = self.model.predict(X_final_scaled)[0]
            
        return "Expensive" if prediction == 1 else "Non-Expensive"
