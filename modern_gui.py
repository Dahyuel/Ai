import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import threading
import matplotlib.pyplot as plt
import seaborn as sns
from models import ModelHandler

# Set appearance mode and default color theme
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class ChatbotApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Spec-to-Price Assistant")
        self.geometry("900x700")
        
        self.handler = ModelHandler()
        self.is_trained = False
        
        self.setup_ui()
        self.after(100, self.start_training)

    def setup_ui(self):
        # Grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0) # Header
        self.grid_rowconfigure(1, weight=1) # Chat Area
        self.grid_rowconfigure(2, weight=0) # Input Area

        # === Header ===
        self.header_frame = ctk.CTkFrame(self, corner_radius=0)
        self.header_frame.grid(row=0, column=0, sticky="ew")
        
        self.header_label = ctk.CTkLabel(self.header_frame, text="Spec-to-Price Assistant", font=ctk.CTkFont(size=20, weight="bold"))
        self.header_label.pack(pady=10, padx=20, side="left")
        
        self.status_label = ctk.CTkLabel(self.header_frame, text="Status: Initializing...", text_color="orange")
        self.status_label.pack(pady=10, padx=20, side="right")

        # === Chat Area ===
        self.chat_frame = ctk.CTkScrollableFrame(self, label_text="Conversation")
        self.chat_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        
        # === Input Area ===
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)
        
        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Type a message...")
        self.entry.grid(row=0, column=0, padx=(10, 10), pady=10, sticky="ew")
        self.entry.bind("<Return>", self.send_message)
        
        self.send_button = ctk.CTkButton(self.input_frame, text="Send", width=60, command=self.send_message)
        self.send_button.grid(row=0, column=1, padx=(0, 5), pady=10)
        
        # Predict Button (Prominent)
        self.predict_btn = ctk.CTkButton(self.input_frame, text="Predict Price", fg_color="green", hover_color="darkgreen", command=self.open_prediction_dialog)
        self.predict_btn.grid(row=0, column=2, padx=(5, 10), pady=10)

    def add_message(self, text, sender="Bot"):
        msg_frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        msg_frame.pack(fill="x", pady=5)
        
        if sender == "Bot":
            color = "gray20"
            anchor = "w"
            justify = "left"
        else:
            color = "#1f538d" # Blueish
            anchor = "e"
            justify = "right"
            
        bubble = ctk.CTkLabel(
            msg_frame, 
            text=text, 
            fg_color=color, 
            corner_radius=10, 
            padx=10, 
            pady=5,
            wraplength=600,
            justify=justify
        )
        bubble.pack(anchor=anchor)
        
        # Scroll to bottom
        self.chat_frame._parent_canvas.yview_moveto(1.0)

    def start_training(self):
        # Silent training
        threading.Thread(target=self._train_thread, daemon=True).start()

    def _train_thread(self):
        try:
            self.handler.load_and_preprocess("Datasets/train.csv")
            acc, _, _ = self.handler.train_random_forest(n_estimators=100)
            self.is_trained = True
            
            self.status_label.configure(text="Status: Ready", text_color="green")
            self.add_message("Hello! I am your Price Prediction Assistant.\n\nI have analyzed the market data and I'm ready to help.\n\nClick 'Predict Price' to check a phone's value!")
        except Exception as e:
            self.status_label.configure(text="Status: Error", text_color="red")
            self.add_message(f"Error during initialization: {e}")

    def send_message(self, event=None):
        text = self.entry.get()
        if not text:
            return
            
        self.add_message(text, sender="User")
        self.entry.delete(0, "end")
        
        self.process_input(text.lower().strip())

    def process_input(self, text):
        if not self.is_trained:
            self.add_message("Please wait a moment, I am getting ready...")
            return

        if "evaluate" in text:
            self.add_message("Evaluating on test set...")
            threading.Thread(target=self._evaluate_thread, daemon=True).start()
            
        elif "predict" in text:
            self.open_prediction_dialog()
            
        elif "plot" in text:
            self.show_plot()
            
        elif "hello" in text or "hi" in text:
            self.add_message("Hello! Click 'Predict Price' to start.")
            
        else:
            self.add_message("I'm here to help you predict phone prices. Just click the 'Predict Price' button!")

    def _evaluate_thread(self):
        try:
            acc, report, cm = self.handler.evaluate_on_test_file("Datasets/test.csv")
            self.add_message(f"Test Set Accuracy: {acc:.4f}")
            self.add_message(f"Classification Report:\n{report}")
            self.last_cm = cm
            self.add_message("Evaluation done. Type 'plot' to see the confusion matrix.")
        except Exception as e:
            self.add_message(f"Error evaluating: {e}")

    def show_plot(self):
        if hasattr(self, 'last_cm'):
            plt.figure(figsize=(8, 6))
            sns.heatmap(self.last_cm, annot=True, fmt='d', cmap='Blues')
            plt.title("Confusion Matrix")
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
        else:
            self.add_message("No evaluation results to plot yet. Run 'evaluate' first.")

    def open_prediction_dialog(self):
        if hasattr(self, 'dialog') and self.dialog.winfo_exists():
            self.dialog.lift()
            return

        self.dialog = ctk.CTkToplevel(self)
        self.dialog.title("Enter Phone Specs")
        self.dialog.geometry("500x800")
        
        # Scrollable frame for inputs
        scroll = ctk.CTkScrollableFrame(self.dialog)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.fields = {}
        
        # === 1. Categorical Features (Dropdowns) ===
        ctk.CTkLabel(scroll, text="Basic Info", font=("Arial", 14, "bold")).pack(pady=(10, 5), anchor="w")
        
        cat_cols = ['brand', 'os_name', 'Processor_Brand', 'Processor_Series', 'Notch_Type']
        for col in cat_cols:
            row = ctk.CTkFrame(scroll, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=col.replace('_', ' ').title(), width=150, anchor="w").pack(side="left", padx=5)
            
            values = ["Unknown"]
            if self.handler.encoders and col in self.handler.encoders:
                values = list(self.handler.encoders[col].classes_)
                # Convert numpy types to strings
                values = [str(v) for v in values]
            
            combo = ctk.CTkComboBox(row, values=values)
            combo.pack(side="right", expand=True, fill="x", padx=5)
            self.fields[col] = ('combo', combo)

        # === 2. Binary Features (Checkboxes) ===
        ctk.CTkLabel(scroll, text="Connectivity & Features", font=("Arial", 14, "bold")).pack(pady=(15, 5), anchor="w")
        
        bin_cols = ['Dual_Sim', '4G', '5G', 'Vo5G', 'NFC', 'IR_Blaster', 'memory_card_support']
        # Grid layout for checkboxes to save space
        bin_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        bin_frame.pack(fill="x", pady=2)
        
        for i, col in enumerate(bin_cols):
            chk = ctk.CTkCheckBox(bin_frame, text=col.replace('_', ' '))
            chk.grid(row=i//2, column=i%2, sticky="w", padx=10, pady=5)
            self.fields[col] = ('check', chk)

        # === 3. Numerical Features (Entries) ===
        ctk.CTkLabel(scroll, text="Technical Specs", font=("Arial", 14, "bold")).pack(pady=(15, 5), anchor="w")
        
        num_cols = [
            'ram_capacity', 'internal_memory', 'battery_capacity', 'screen_size', 
            'num_rear_cameras', 'num_front_cameras', 'primary_camera_rear', 'primary_camera_front',
            'resolution_height', 'resolution_width', 'refresh_rate', 'os_version'
        ]
        
        for col in num_cols:
            row = ctk.CTkFrame(scroll, fg_color="transparent")
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=col.replace('_', ' ').title(), width=150, anchor="w").pack(side="left", padx=5)
            entry = ctk.CTkEntry(row)
            entry.pack(side="right", expand=True, fill="x", padx=5)
            self.fields[col] = ('entry', entry)

        def submit():
            data = {}
            try:
                # Collect Categorical
                for col in cat_cols:
                    type_, widget = self.fields[col]
                    data[col] = widget.get()
                
                # Collect Binary
                for col in bin_cols:
                    type_, widget = self.fields[col]
                    # Map 1/0 to Yes/No for preprocess
                    data[col] = "Yes" if widget.get() == 1 else "No"
                
                # Collect Numerical
                for col in num_cols:
                    type_, widget = self.fields[col]
                    val = widget.get()
                    if not val:
                        # Default values
                        if col == 'os_version': val = "10"
                        else: val = "0"
                    
                    # os_version is string but needs to be parsed by regex in preprocess
                    if col == 'os_version':
                        data[col] = str(val)
                    else:
                        data[col] = float(val)
                
                # Close dialog
                self.dialog.destroy()
                
                # Predict
                result = self.handler.predict_single(data)
                self.add_message(f"Prediction Result: {result}")
                self.add_message(f"Specs Summary:\nBrand: {data.get('brand')}\nRAM: {data.get('ram_capacity')}GB\nStorage: {data.get('internal_memory')}GB\n5G: {data.get('5G')}")
                
            except ValueError:
                messagebox.showerror("Error", "Please enter valid numbers for technical specs.")
            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {e}")

        ctk.CTkButton(self.dialog, text="Predict Price", fg_color="green", hover_color="darkgreen", command=submit).pack(pady=20)

if __name__ == "__main__":
    app = ChatbotApp()
    app.mainloop()
