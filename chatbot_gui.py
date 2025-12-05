import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import threading
import os
import sys

# Add current directory to path for importing preprocessing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import preprocess


class ModernPhonePricePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("📱 Phone Price Predictor - AI Chatbot")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Modern dark theme colors
        self.colors = {
            'bg_dark': '#1a1a2e',
            'bg_medium': '#16213e',
            'bg_light': '#0f3460',
            'accent': '#e94560',
            'accent_hover': '#ff6b6b',
            'text_primary': '#ffffff',
            'text_secondary': '#a0a0a0',
            'user_bubble': '#0f3460',
            'bot_bubble': '#2d3561',
            'input_bg': '#252a41',
            'border': '#3a3f5c',
            'success': '#00d26a',
            'warning': '#ffc107'
        }
        
        self.root.configure(bg=self.colors['bg_dark'])
        
        # Model variables
        self.model = None
        self.encoders = None
        self.feature_cols = None
        self.scaler = None
        self.unique_values = {}
        
        # Configure styles
        self.setup_styles()
        
        # Create main layout
        self.create_layout()
        
        # Start model training in background
        self.add_bot_message("👋 Welcome to the Phone Price Predictor!")
        self.add_bot_message("🔄 Loading the AI model... Please wait.")
        threading.Thread(target=self.train_model, daemon=True).start()
    
    def setup_styles(self):
        """Configure ttk styles for modern look"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure main styles
        style.configure('Dark.TFrame', background=self.colors['bg_dark'])
        style.configure('Medium.TFrame', background=self.colors['bg_medium'])
        style.configure('Light.TFrame', background=self.colors['bg_light'])
        
        style.configure('Title.TLabel',
                       background=self.colors['bg_dark'],
                       foreground=self.colors['text_primary'],
                       font=('Segoe UI', 24, 'bold'))
        
        style.configure('Subtitle.TLabel',
                       background=self.colors['bg_dark'],
                       foreground=self.colors['text_secondary'],
                       font=('Segoe UI', 11))
        
        style.configure('Section.TLabel',
                       background=self.colors['bg_medium'],
                       foreground=self.colors['accent'],
                       font=('Segoe UI', 12, 'bold'))
        
        style.configure('Input.TLabel',
                       background=self.colors['bg_medium'],
                       foreground=self.colors['text_primary'],
                       font=('Segoe UI', 10))
        
        style.configure('Dark.TCheckbutton',
                       background=self.colors['bg_medium'],
                       foreground=self.colors['text_primary'],
                       font=('Segoe UI', 10))
        
        style.map('Dark.TCheckbutton',
                 background=[('active', self.colors['bg_medium'])])
        
        style.configure('Accent.TButton',
                       background=self.colors['accent'],
                       foreground=self.colors['text_primary'],
                       font=('Segoe UI', 12, 'bold'),
                       padding=(20, 12))
        
        style.map('Accent.TButton',
                 background=[('active', self.colors['accent_hover'])])
        
        style.configure('TCombobox',
                       fieldbackground=self.colors['input_bg'],
                       background=self.colors['input_bg'],
                       foreground=self.colors['text_primary'],
                       arrowcolor=self.colors['text_primary'])
        
        style.map('TCombobox',
                 fieldbackground=[('readonly', self.colors['input_bg'])],
                 selectbackground=[('readonly', self.colors['accent'])])
    
    def create_layout(self):
        """Create the main application layout"""
        # Main container with padding
        main_container = ttk.Frame(self.root, style='Dark.TFrame')
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ttk.Frame(main_container, style='Dark.TFrame')
        header_frame.pack(fill='x', pady=(0, 20))
        
        title_label = ttk.Label(header_frame, text="📱 Phone Price Predictor",
                               style='Title.TLabel')
        title_label.pack(side='left')
        
        subtitle_label = ttk.Label(header_frame,
                                  text="AI-Powered Price Category Prediction",
                                  style='Subtitle.TLabel')
        subtitle_label.pack(side='left', padx=(20, 0), pady=(10, 0))
        
        # Content area - split into left (inputs) and right (chat)
        content_frame = ttk.Frame(main_container, style='Dark.TFrame')
        content_frame.pack(fill='both', expand=True)
        
        # Left panel - Inputs (scrollable)
        self.create_input_panel(content_frame)
        
        # Right panel - Chat
        self.create_chat_panel(content_frame)
    
    def create_input_panel(self, parent):
        """Create the input panel with all fields"""
        # Container for inputs
        input_container = ttk.Frame(parent, style='Medium.TFrame')
        input_container.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Canvas and scrollbar for scrolling
        canvas = tk.Canvas(input_container, bg=self.colors['bg_medium'],
                          highlightthickness=0)
        scrollbar = ttk.Scrollbar(input_container, orient='vertical',
                                 command=canvas.yview)
        
        self.input_frame = ttk.Frame(canvas, style='Medium.TFrame')
        
        self.input_frame.bind('<Configure>',
                             lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        
        canvas.create_window((0, 0), window=self.input_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), 'units')
        
        canvas.bind_all('<MouseWheel>', on_mousewheel)
        
        scrollbar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        
        # Create input sections
        self.input_vars = {}
        
        # Binary Options Section
        self.create_section_header("🔘 Binary Options")
        self.create_binary_inputs()
        
        # Brand & Processor Section
        self.create_section_header("🏭 Brand & Processor")
        self.create_brand_processor_inputs()
        
        # Display Section
        self.create_section_header("📺 Display")
        self.create_display_inputs()
        
        # Performance Section
        self.create_section_header("⚡ Performance")
        self.create_performance_inputs()
        
        # Camera Section
        self.create_section_header("📷 Camera")
        self.create_camera_inputs()
        
        # Battery & Storage Section
        self.create_section_header("🔋 Battery & Storage")
        self.create_battery_storage_inputs()
        
        # OS Section
        self.create_section_header("💻 Operating System")
        self.create_os_inputs()
        
        # Predict Button
        self.create_predict_button()
    
    def create_section_header(self, text):
        """Create a section header"""
        header = ttk.Label(self.input_frame, text=text, style='Section.TLabel')
        header.pack(fill='x', pady=(15, 5), padx=10)
        
        # Separator line
        separator = tk.Frame(self.input_frame, height=1,
                            bg=self.colors['border'])
        separator.pack(fill='x', padx=10, pady=(0, 10))
    
    def create_binary_inputs(self):
        """Create binary (Yes/No) input fields"""
        binary_fields = [
            ('Dual_Sim', 'Dual SIM'),
            ('4G', '4G Support'),
            ('5G', '5G Support'),
            ('Vo5G', 'Voice over 5G'),
            ('NFC', 'NFC'),
            ('IR_Blaster', 'IR Blaster'),
            ('memory_card_support', 'Memory Card Support')
        ]
        
        frame = ttk.Frame(self.input_frame, style='Medium.TFrame')
        frame.pack(fill='x', padx=10, pady=5)
        
        for i, (field_name, display_name) in enumerate(binary_fields):
            var = tk.BooleanVar(value=True)
            self.input_vars[field_name] = var
            
            cb = ttk.Checkbutton(frame, text=display_name, variable=var,
                                style='Dark.TCheckbutton')
            cb.grid(row=i//3, column=i%3, padx=10, pady=5, sticky='w')
    
    def create_brand_processor_inputs(self):
        """Create brand and processor inputs"""
        frame = ttk.Frame(self.input_frame, style='Medium.TFrame')
        frame.pack(fill='x', padx=10, pady=5)
        
        # Brand
        self.create_combobox(frame, 'brand', 'Brand', 0, 0,
                            ['iQOO', 'Samsung', 'Apple', 'OnePlus', 'Xiaomi',
                             'Realme', 'Vivo', 'OPPO', 'Motorola', 'Google',
                             'Nokia', 'Honor', 'Poco', 'Redmi', 'Huawei',
                             'Asus', 'Sony', 'LG', 'Lenovo', 'Nothing'])
        
        # Processor Brand
        self.create_combobox(frame, 'Processor_Brand', 'Processor Brand', 0, 1,
                            ['Snapdragon', 'MediaTek', 'Exynos', 'Apple',
                             'Google Tensor', 'Unisoc', 'Kirin', 'Dimensity'])
        
        # Processor Series
        self.create_entry(frame, 'Processor_Series', 'Processor Series', 1, 0, '870')
        
        # Rating
        self.create_entry(frame, 'rating', 'Rating (0-100)', 1, 1, '85')
    
    def create_display_inputs(self):
        """Create display-related inputs"""
        frame = ttk.Frame(self.input_frame, style='Medium.TFrame')
        frame.pack(fill='x', padx=10, pady=5)
        
        # Screen Size
        self.create_entry(frame, 'Screen_Size', 'Screen Size (inches)', 0, 0, '6.5')
        
        # Resolution Width
        self.create_entry(frame, 'Resolution_Width', 'Resolution Width', 0, 1, '1080')
        
        # Resolution Height
        self.create_entry(frame, 'Resolution_Height', 'Resolution Height', 1, 0, '2400')
        
        # Refresh Rate
        self.create_entry(frame, 'Refresh_Rate', 'Refresh Rate (Hz)', 1, 1, '120')
        
        # Notch Type
        self.create_combobox(frame, 'Notch_Type', 'Notch Type', 2, 0,
                            ['Punch Hole', 'Water Drop Notch', 'No Notch',
                             'Dual Punch Hole', 'Large Notch', 'Dynamic Island'],
                            span=2)
    
    def create_performance_inputs(self):
        """Create performance-related inputs"""
        frame = ttk.Frame(self.input_frame, style='Medium.TFrame')
        frame.pack(fill='x', padx=10, pady=5)
        
        # Core Count
        self.create_entry(frame, 'Core_Count', 'CPU Cores', 0, 0, '8')
        
        # Clock Speed
        self.create_entry(frame, 'Clock_Speed_GHz', 'Clock Speed (GHz)', 0, 1, '3.0')
        
        # RAM
        self.create_entry(frame, 'RAM Size GB', 'RAM (GB)', 1, 0, '8')
        
        # Storage
        self.create_entry(frame, 'Storage Size GB', 'Storage (GB)', 1, 1, '256')
    
    def create_camera_inputs(self):
        """Create camera-related inputs"""
        frame = ttk.Frame(self.input_frame, style='Medium.TFrame')
        frame.pack(fill='x', padx=10, pady=5)
        
        # Primary Rear Camera
        self.create_entry(frame, 'primary_rear_camera_mp', 'Main Camera (MP)', 0, 0, '48')
        
        # Number of Rear Cameras
        self.create_entry(frame, 'num_rear_cameras', 'Rear Cameras Count', 0, 1, '3')
        
        # Primary Front Camera
        self.create_entry(frame, 'primary_front_camera_mp', 'Front Camera (MP)', 1, 0, '16')
        
        # Number of Front Cameras
        self.create_entry(frame, 'num_front_cameras', 'Front Cameras Count', 1, 1, '1')
    
    def create_battery_storage_inputs(self):
        """Create battery and storage inputs"""
        frame = ttk.Frame(self.input_frame, style='Medium.TFrame')
        frame.pack(fill='x', padx=10, pady=5)
        
        # Battery Capacity
        self.create_entry(frame, 'battery_capacity', 'Battery (mAh)', 0, 0, '5000')
        
        # Fast Charging
        self.create_entry(frame, 'fast_charging_power', 'Fast Charging (W)', 0, 1, '65')
        
        # Memory Card Size
        self.create_entry(frame, 'memory_card_size', 'Max SD Card (GB)', 1, 0, '512')
    
    def create_os_inputs(self):
        """Create OS-related inputs"""
        frame = ttk.Frame(self.input_frame, style='Medium.TFrame')
        frame.pack(fill='x', padx=10, pady=5)
        
        # OS Name
        self.create_combobox(frame, 'os_name', 'Operating System', 0, 0,
                            ['Android', 'iOS', 'HarmonyOS', 'EMUI'])
        
        # OS Version
        self.create_entry(frame, 'os_version', 'OS Version', 0, 1, 'v14')
    
    def create_entry(self, parent, name, label, row, col, default=''):
        """Create a labeled entry field"""
        container = ttk.Frame(parent, style='Medium.TFrame')
        container.grid(row=row, column=col, padx=10, pady=5, sticky='ew')
        
        parent.columnconfigure(col, weight=1)
        
        lbl = ttk.Label(container, text=label, style='Input.TLabel')
        lbl.pack(anchor='w')
        
        var = tk.StringVar(value=default)
        self.input_vars[name] = var
        
        entry = tk.Entry(container, textvariable=var,
                        bg=self.colors['input_bg'],
                        fg=self.colors['text_primary'],
                        insertbackground=self.colors['text_primary'],
                        relief='flat',
                        font=('Segoe UI', 10))
        entry.pack(fill='x', pady=(5, 0), ipady=8)
    
    def create_combobox(self, parent, name, label, row, col, values, span=1):
        """Create a labeled combobox"""
        container = ttk.Frame(parent, style='Medium.TFrame')
        container.grid(row=row, column=col, columnspan=span,
                      padx=10, pady=5, sticky='ew')
        
        parent.columnconfigure(col, weight=1)
        
        lbl = ttk.Label(container, text=label, style='Input.TLabel')
        lbl.pack(anchor='w')
        
        var = tk.StringVar(value=values[0] if values else '')
        self.input_vars[name] = var
        
        combo = ttk.Combobox(container, textvariable=var, values=values,
                            state='readonly', font=('Segoe UI', 10))
        combo.pack(fill='x', pady=(5, 0))
    
    def create_predict_button(self):
        """Create the predict button"""
        btn_frame = ttk.Frame(self.input_frame, style='Medium.TFrame')
        btn_frame.pack(fill='x', padx=10, pady=20)
        
        predict_btn = tk.Button(btn_frame, text="🔮 Predict Price Category",
                               bg=self.colors['accent'],
                               fg=self.colors['text_primary'],
                               activebackground=self.colors['accent_hover'],
                               activeforeground=self.colors['text_primary'],
                               font=('Segoe UI', 14, 'bold'),
                               relief='flat',
                               cursor='hand2',
                               command=self.predict_price)
        predict_btn.pack(fill='x', ipady=15)
        
        # Clear button
        clear_btn = tk.Button(btn_frame, text="🗑️ Clear All",
                             bg=self.colors['bg_light'],
                             fg=self.colors['text_primary'],
                             activebackground=self.colors['border'],
                             activeforeground=self.colors['text_primary'],
                             font=('Segoe UI', 11),
                             relief='flat',
                             cursor='hand2',
                             command=self.clear_inputs)
        clear_btn.pack(fill='x', ipady=10, pady=(10, 0))
    
    def create_chat_panel(self, parent):
        """Create the chat panel"""
        chat_container = ttk.Frame(parent, style='Medium.TFrame')
        chat_container.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Chat header
        header = ttk.Label(chat_container, text="💬 Prediction History",
                          style='Section.TLabel')
        header.pack(fill='x', padx=10, pady=10)
        
        # Chat display
        self.chat_display = tk.Text(chat_container,
                                   bg=self.colors['bg_dark'],
                                   fg=self.colors['text_primary'],
                                   font=('Segoe UI', 11),
                                   relief='flat',
                                   wrap='word',
                                   padx=15,
                                   pady=15,
                                   state='disabled')
        self.chat_display.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Configure text tags for styling
        self.chat_display.tag_configure('bot',
                                        foreground=self.colors['accent'],
                                        font=('Segoe UI', 11, 'bold'))
        self.chat_display.tag_configure('user',
                                        foreground=self.colors['success'],
                                        font=('Segoe UI', 11, 'bold'))
        self.chat_display.tag_configure('message',
                                        foreground=self.colors['text_primary'])
        self.chat_display.tag_configure('expensive',
                                        foreground='#ff6b6b',
                                        font=('Segoe UI', 12, 'bold'))
        self.chat_display.tag_configure('non_expensive',
                                        foreground='#00d26a',
                                        font=('Segoe UI', 12, 'bold'))
        self.chat_display.tag_configure('timestamp',
                                        foreground=self.colors['text_secondary'],
                                        font=('Segoe UI', 9))
    
    def add_bot_message(self, message):
        """Add a bot message to the chat"""
        self.chat_display.config(state='normal')
        self.chat_display.insert('end', '\n🤖 Bot: ', 'bot')
        self.chat_display.insert('end', message + '\n', 'message')
        self.chat_display.config(state='disabled')
        self.chat_display.see('end')
    
    def add_user_message(self, message):
        """Add a user message to the chat"""
        self.chat_display.config(state='normal')
        self.chat_display.insert('end', '\n👤 You: ', 'user')
        self.chat_display.insert('end', message + '\n', 'message')
        self.chat_display.config(state='disabled')
        self.chat_display.see('end')
    
    def add_prediction_result(self, prediction, confidence=None):
        """Add a prediction result to the chat"""
        self.chat_display.config(state='normal')
        self.chat_display.insert('end', '\n🎯 Prediction: ', 'bot')
        
        if prediction == 1:
            self.chat_display.insert('end', '💰 EXPENSIVE\n', 'expensive')
        else:
            self.chat_display.insert('end', '💵 NON-EXPENSIVE\n', 'non_expensive')
        
        if confidence is not None:
            self.chat_display.insert('end', f'   Confidence: {confidence:.1%}\n', 'message')
        
        self.chat_display.config(state='disabled')
        self.chat_display.see('end')
    
    def train_model(self):
        """Train the Random Forest model in background"""
        try:
            # Get the directory of this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(script_dir, 'Datasets', 'train.csv')
            
            # Load data
            df = pd.read_csv(data_path)
            
            # Store unique values for validation
            self.unique_values = {
                'brand': df['brand'].unique().tolist(),
                'Processor_Brand': df['Processor_Brand'].unique().tolist(),
                'Processor_Series': df['Processor_Series'].unique().tolist(),
                'Notch_Type': df['Notch_Type'].unique().tolist(),
                'os_name': df['os_name'].unique().tolist(),
                'os_version': df['os_version'].unique().tolist()
            }
            
            # Preprocess train set (fit encoders)
            df_processed, self.encoders = preprocess(df, fit=True)
            
            # Select features
            self.feature_cols = [
                col for col in df_processed.columns 
                if col not in ['price', 'price_encoded'] and df_processed[col].dtype != 'object'
            ]
            
            X = df_processed[self.feature_cols]
            y = df_processed['price_encoded']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                max_depth=20,
                min_samples_split=5
            )
            self.model.fit(X_train, y_train)
            
            # Update chat on main thread
            self.root.after(0, lambda: self.add_bot_message(
                "✅ AI Model loaded successfully! Enter phone specs and click 'Predict Price Category'."
            ))
            
        except Exception as e:
            self.root.after(0, lambda: self.add_bot_message(
                f"❌ Error loading model: {str(e)}"
            ))
    
    def get_input_values(self):
        """Get all input values as a dictionary"""
        values = {}
        
        for name, var in self.input_vars.items():
            if isinstance(var, tk.BooleanVar):
                values[name] = 'Yes' if var.get() else 'No'
            else:
                values[name] = var.get()
        
        return values
    
    def predict_price(self):
        """Make a price prediction based on inputs"""
        if self.model is None:
            self.add_bot_message("⏳ Please wait, the model is still loading...")
            return
        
        try:
            # Get input values
            values = self.get_input_values()
            
            # Create a summary message
            summary = f"Brand: {values.get('brand', 'N/A')}, "
            summary += f"RAM: {values.get('RAM Size GB', 'N/A')}GB, "
            summary += f"Storage: {values.get('Storage Size GB', 'N/A')}GB"
            self.add_user_message(f"Predicting price for: {summary}")
            
            # Create input DataFrame
            input_data = {
                'price': ['unknown'],  # Placeholder, will be removed
                'rating': [float(values.get('rating', 85))],
                'Dual_Sim': [values.get('Dual_Sim', 'Yes')],
                '4G': [values.get('4G', 'Yes')],
                '5G': [values.get('5G', 'Yes')],
                'Vo5G': [values.get('Vo5G', 'No')],
                'NFC': [values.get('NFC', 'Yes')],
                'IR_Blaster': [values.get('IR_Blaster', 'No')],
                'Processor_Brand': [values.get('Processor_Brand', 'Snapdragon')],
                'Processor_Series': [values.get('Processor_Series', '870')],
                'Core_Count': [float(values.get('Core_Count', 8))],
                'Clock_Speed_GHz': [float(values.get('Clock_Speed_GHz', 3.0))],
                'RAM Size GB': [float(values.get('RAM Size GB', 8))],
                'Storage Size GB': [float(values.get('Storage Size GB', 256))],
                'battery_capacity': [float(values.get('battery_capacity', 5000))],
                'fast_charging_power': [int(values.get('fast_charging_power', 65))],
                'Screen_Size': [float(values.get('Screen_Size', 6.5))],
                'Resolution_Width': [float(values.get('Resolution_Width', 1080))],
                'Resolution_Height': [float(values.get('Resolution_Height', 2400))],
                'Refresh_Rate': [float(values.get('Refresh_Rate', 120))],
                'Notch_Type': [values.get('Notch_Type', 'Punch Hole')],
                'primary_rear_camera_mp': [float(values.get('primary_rear_camera_mp', 48))],
                'num_rear_cameras': [float(values.get('num_rear_cameras', 3))],
                'primary_front_camera_mp': [float(values.get('primary_front_camera_mp', 16))],
                'num_front_cameras': [float(values.get('num_front_cameras', 1))],
                'memory_card_support': [values.get('memory_card_support', 'Yes')],
                'memory_card_size': [f"{values.get('memory_card_size', '512')} GB"],
                'os_name': [values.get('os_name', 'Android')],
                'os_version': [values.get('os_version', 'v14')],
                'brand': [values.get('brand', 'Samsung')]
            }
            
            input_df = pd.DataFrame(input_data)
            
            # Remove price column for preprocessing
            input_df_features = input_df.drop(columns=['price'])
            
            # Preprocess using existing encoders
            processed_df, _ = preprocess(
                input_df_features,
                label_encoders=self.encoders,
                fit=False
            )
            
            # Ensure all feature columns exist
            for col in self.feature_cols:
                if col not in processed_df.columns:
                    processed_df[col] = 0
            
            # Get features in correct order
            X_pred = processed_df[self.feature_cols]
            
            # Make prediction
            prediction = self.model.predict(X_pred)[0]
            
            # Get prediction probability
            proba = self.model.predict_proba(X_pred)[0]
            confidence = max(proba)
            
            # Display result
            self.add_prediction_result(prediction, confidence)
            
        except Exception as e:
            self.add_bot_message(f"❌ Error making prediction: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def clear_inputs(self):
        """Clear all input fields to defaults"""
        defaults = {
            'Dual_Sim': True, '4G': True, '5G': True, 'Vo5G': False,
            'NFC': True, 'IR_Blaster': False, 'memory_card_support': True,
            'brand': 'Samsung', 'Processor_Brand': 'Snapdragon',
            'Processor_Series': '870', 'rating': '85',
            'Screen_Size': '6.5', 'Resolution_Width': '1080',
            'Resolution_Height': '2400', 'Refresh_Rate': '120',
            'Notch_Type': 'Punch Hole', 'Core_Count': '8',
            'Clock_Speed_GHz': '3.0', 'RAM Size GB': '8',
            'Storage Size GB': '256', 'primary_rear_camera_mp': '48',
            'num_rear_cameras': '3', 'primary_front_camera_mp': '16',
            'num_front_cameras': '1', 'battery_capacity': '5000',
            'fast_charging_power': '65', 'memory_card_size': '512',
            'os_name': 'Android', 'os_version': 'v14'
        }
        
        for name, default in defaults.items():
            if name in self.input_vars:
                if isinstance(self.input_vars[name], tk.BooleanVar):
                    self.input_vars[name].set(default)
                else:
                    self.input_vars[name].set(str(default))
        
        self.add_bot_message("🔄 All inputs have been reset to defaults.")


def main():
    root = tk.Tk()
    app = ModernPhonePricePredictor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
