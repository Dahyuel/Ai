import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import os
from models import ModelHandler

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Spec-to-Price Predictor")
        self.root.geometry("1200x800")
        
        self.handler = ModelHandler()
        self.train_path = tk.StringVar(value="Datasets/train.csv")
        self.test_path = tk.StringVar(value="Datasets/test.csv")
        
        self.setup_ui()
        
    def setup_ui(self):
        # === Left Control Panel ===
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Data Selection
        ttk.Label(control_frame, text="Data Selection", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        
        ttk.Label(control_frame, text="Train CSV:").pack(anchor="w")
        train_frame = ttk.Frame(control_frame)
        train_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(train_frame, textvariable=self.train_path, width=25).pack(side=tk.LEFT)
        ttk.Button(train_frame, text="Browse", command=lambda: self.browse_file(self.train_path)).pack(side=tk.LEFT)
        
        ttk.Label(control_frame, text="Test CSV:").pack(anchor="w")
        test_frame = ttk.Frame(control_frame)
        test_frame.pack(fill=tk.X, pady=2)
        ttk.Entry(test_frame, textvariable=self.test_path, width=25).pack(side=tk.LEFT)
        ttk.Button(test_frame, text="Browse", command=lambda: self.browse_file(self.test_path)).pack(side=tk.LEFT)
        
        ttk.Button(control_frame, text="Load Data", command=self.load_data).pack(fill=tk.X, pady=10)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Model Selection
        ttk.Label(control_frame, text="Model Configuration", font=("Arial", 12, "bold")).pack(pady=5, anchor="w")
        
        self.model_var = tk.StringVar(value="KNN")
        ttk.Label(control_frame, text="Select Model:").pack(anchor="w")
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, values=["KNN", "Logistic Regression", "Random Forest"])
        model_combo.pack(fill=tk.X, pady=2)
        model_combo.bind("<<ComboboxSelected>>", self.update_params)
        
        # Dynamic Parameters Frame
        self.param_frame = ttk.LabelFrame(control_frame, text="Hyperparameters", padding="5")
        self.param_frame.pack(fill=tk.X, pady=10)
        self.param_vars = {}
        self.update_params()
        
        # Actions
        ttk.Button(control_frame, text="Train Model", command=self.train_model).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="Evaluate on Test Set", command=self.evaluate_test).pack(fill=tk.X, pady=5)
        
        # === Right Results Panel ===
        results_frame = ttk.Frame(self.root, padding="10")
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Text Output Tab
        self.text_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.text_tab, text="Metrics & Logs")
        
        self.log_text = tk.Text(self.text_tab, wrap=tk.WORD, font=("Consolas", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Visualization Tab
        self.plot_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_tab, text="Visualizations")
        
        self.figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def browse_file(self, var):
        filename = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if filename:
            var.set(filename)
            
    def update_params(self, event=None):
        # Clear existing params
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_vars = {}
        
        model = self.model_var.get()
        if model == "KNN":
            ttk.Label(self.param_frame, text="Neighbors (k):").pack(anchor="w")
            var = tk.IntVar(value=5)
            ttk.Entry(self.param_frame, textvariable=var).pack(fill=tk.X)
            self.param_vars['k'] = var
            
        elif model == "Logistic Regression":
            ttk.Label(self.param_frame, text="Max Iterations:").pack(anchor="w")
            var = tk.IntVar(value=300)
            ttk.Entry(self.param_frame, textvariable=var).pack(fill=tk.X)
            self.param_vars['max_iter'] = var
            
        elif model == "Random Forest":
            ttk.Label(self.param_frame, text="N Estimators:").pack(anchor="w")
            var1 = tk.IntVar(value=200)
            ttk.Entry(self.param_frame, textvariable=var1).pack(fill=tk.X)
            self.param_vars['n_estimators'] = var1
            
            ttk.Label(self.param_frame, text="Max Depth:").pack(anchor="w")
            var2 = tk.IntVar(value=20)
            ttk.Entry(self.param_frame, textvariable=var2).pack(fill=tk.X)
            self.param_vars['max_depth'] = var2

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def load_data(self):
        try:
            msg = self.handler.load_and_preprocess(self.train_path.get())
            self.log(f"[INFO] {msg}")
            messagebox.showinfo("Success", "Data loaded successfully!")
        except Exception as e:
            self.log(f"[ERROR] Failed to load data: {e}")
            messagebox.showerror("Error", str(e))

    def train_model(self):
        try:
            model_name = self.model_var.get()
            self.log(f"\n[INFO] Training {model_name}...")
            
            if model_name == "KNN":
                k = self.param_vars['k'].get()
                acc, report, cm = self.handler.train_knn(k=k)
            elif model_name == "Logistic Regression":
                max_iter = self.param_vars['max_iter'].get()
                acc, report, cm = self.handler.train_logistic_regression(max_iter=max_iter)
            elif model_name == "Random Forest":
                n_est = self.param_vars['n_estimators'].get()
                depth = self.param_vars['max_depth'].get()
                acc, report, cm = self.handler.train_random_forest(n_estimators=n_est, max_depth=depth)
            
            self.log(f"Training Accuracy (Validation Split): {acc:.4f}")
            self.log("Classification Report:\n" + report)
            
            self.plot_confusion_matrix(cm, f"Confusion Matrix - {model_name} (Val)")
            
            # If RF, plot feature importance
            if model_name == "Random Forest":
                names, imps = self.handler.get_feature_importance()
                self.plot_feature_importance(names, imps)
                
        except Exception as e:
            self.log(f"[ERROR] Training failed: {e}")
            messagebox.showerror("Error", str(e))

    def evaluate_test(self):
        try:
            self.log(f"\n[INFO] Evaluating on Test Set...")
            acc, report, cm = self.handler.evaluate_on_test_file(self.test_path.get())
            
            self.log(f"Test Set Accuracy: {acc:.4f}")
            self.log("Classification Report:\n" + report)
            
            self.plot_confusion_matrix(cm, f"Confusion Matrix - {self.model_var.get()} (Test)")
            
        except Exception as e:
            self.log(f"[ERROR] Evaluation failed: {e}")
            messagebox.showerror("Error", str(e))

    def plot_confusion_matrix(self, cm, title):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        self.canvas.draw()
        self.notebook.select(self.plot_tab)

    def plot_feature_importance(self, names, importances):
        # Create a new window for this since the main plot area is for CM
        top_n = 15
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 6))
        plt.title("Top 15 Feature Importances")
        plt.barh(range(top_n), importances[indices], align="center")
        plt.yticks(range(top_n), [names[i] for i in indices])
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Ensure we are in the right directory or paths are absolute
    # For this script, we assume it's run from the project root or paths are relative to it
    root = tk.Tk()
    app = App(root)
    root.mainloop()
