import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sys

class JobFitApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Job Fit Prediction Tool')
        self.geometry('600x400')
        self.resizable(False, False)
        
        # Variables for file paths
        self.training_file = r'C:\Users\employee_attrition_dataset.csv'
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        
        # Setup UI
        self.create_widgets()
        self.create_menu()
        
        # Model placeholder
        self.clf = None
        self.X_train_encoded = None
        
    def create_widgets(self):
        # Labels and file path displays
        ttk.Label(self, text='Training Data used (fixed):').pack(anchor='w', padx=10, pady=(10,0))
        ttk.Label(self, text=self.training_file).pack(anchor='w', padx=20)
        
        ttk.Label(self, text='Input CSV for Prediction:').pack(anchor='w', padx=10, pady=(20,0))
        frame_input = ttk.Frame(self)
        frame_input.pack(fill='x', padx=10)
        ttk.Entry(frame_input, textvariable=self.input_file, width=60).pack(side='left', padx=(0,5))
        ttk.Button(frame_input, text='Browse...', command=self.select_input).pack(side='left')
        
        ttk.Label(self, text='Output Excel File:').pack(anchor='w', padx=10, pady=(20,0))
        frame_output = ttk.Frame(self)
        frame_output.pack(fill='x', padx=10)
        ttk.Entry(frame_output, textvariable=self.output_file, width=60).pack(side='left', padx=(0,5))
        ttk.Button(frame_output, text='Browse...', command=self.select_output).pack(side='left')
        
        # Buttons: Run, Clear, Exit
        frame_btn = ttk.Frame(self)
        frame_btn.pack(pady=30)
        
        ttk.Button(frame_btn, text='Run Prediction', command=self.run_prediction).pack(side='left', padx=10)
        ttk.Button(frame_btn, text='Clear', command=self.clear_fields).pack(side='left', padx=10)
        ttk.Button(frame_btn, text='Exit', command=self.exit_app).pack(side='left', padx=10)
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set('Ready')
        ttk.Label(self, textvariable=self.status_var, foreground='blue').pack(side='bottom', fill='x', pady=5)
        
    def create_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label='Select Input CSV', command=self.select_input)
        file_menu.add_command(label='Select Output Excel', command=self.select_output)
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self.exit_app)
        menubar.add_cascade(label='File', menu=file_menu)
        
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label='About', command=self.show_about)
        menubar.add_cascade(label='Help', menu=help_menu)
        
        self.config(menu=menubar)
        
    def select_input(self):
        filename = filedialog.askopenfilename(title='Select Input CSV File', filetypes=[('CSV Files', '*.csv')])
        if filename:
            self.input_file.set(filename)
            self.update_status(f'Selected input file: {filename}')
        
    def select_output(self):
        filename = filedialog.asksaveasfilename(defaultextension='.xlsx', filetypes=[('Excel Files', '*.xlsx')], title='Save Output Excel As')
        if filename:
            self.output_file.set(filename)
            self.update_status(f'Selected output file: {filename}')
    
    def clear_fields(self):
        self.input_file.set('')
        self.output_file.set('')
        self.update_status('Cleared input and output file selections.')
    
    def exit_app(self):
        self.destroy()
        sys.exit()
        
    def update_status(self, message):
        self.status_var.set(message)
        self.update_idletasks()
    
    def check_columns(self, df, required_cols, desc):
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {desc}: {', '.join(missing)}")

    def run_prediction(self):
        try:
            self.update_status('Loading training data...')
            train_data = pd.read_csv(self.training_file)

            # Required columns
            train_required_cols = ['   ', '  ', '   ']   #include required columns

            self.check_columns(train_data, train_required_cols, 'Training Data')

            X = train_data.drop(['Employee_ID', 'Attrition'], axis=1)
            y = train_data['Attrition']

            self.update_status('Encoding features...')
            X_encoded = pd.get_dummies(X)

            self.update_status('Splitting data and training model...')
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

            self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
            self.clf.fit(X_train, y_train)

            self.update_status('Evaluating model...')
            y_val_pred = self.clf.predict(X_val)
            acc = accuracy_score(y_val, y_val_pred)
            c_report = classification_report(y_val, y_val_pred, zero_division=0)

            messagebox.showinfo('Model Evaluation',
                                f'Validation Accuracy: {acc:.4f}\n\nClassification Report:\n{c_report}')

            # Process prediction input file
            input_path = self.input_file.get()
            if not input_path:
                messagebox.showwarning('Input Required', 'Please select an input CSV file.')
                return
            self.update_status('Loading input CSV file for prediction...')
            input_df = pd.read_csv(input_path)

            self.check_columns(input_df, list(X.columns), 'Input Prediction Data')

            self.update_status('Encoding input data...')
            input_encoded = pd.get_dummies(input_df)
            input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

            self.update_status('Running prediction...')
            preds = self.clf.predict(input_encoded)
            input_df['Prediction'] = preds
            input_df['Prediction_Label'] = input_df['Prediction'].map({1: 'Yes', 0: 'No'})

            # Save output
            output_path = self.output_file.get()
            if not output_path:
                messagebox.showwarning('Output Required', 'Please select an output Excel file path.')
                return
            self.update_status(f'Saving predictions to {output_path}...')
            input_df.to_excel(output_path, index=False)

            self.update_status('Prediction complete.')
            messagebox.showinfo('Success', f'Predictions saved successfully to:\n{output_path}')

        except Exception as e:
            messagebox.showerror('Error', str(e))
            self.update_status('Error occurred. See message box.')

    def show_about(self):
        messagebox.showinfo("About", "Job Fit Prediction Tool\nVersion 1.0\nDeveloped by Your Name")

if __name__ == '__main__':
    app = JobFitApp()
    app.mainloop()
