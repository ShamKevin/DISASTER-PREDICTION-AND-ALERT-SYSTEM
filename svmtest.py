import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import cv2
from sklearn.svm import SVC
import joblib
import numpy as np
import os
import os
import urllib.request
# Load your pre-trained SVM model
svm_model = joblib.load('SVM_model.pkl')  # Replace with the correct path to your SVM_model.pkl
data_dir = "data"
class_names = os.listdir(data_dir)

# Function to classify an image
def classify_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and preprocess the selected image
        img = cv2.imread(file_path)
        img = cv2.medianBlur(img, 1)
        img = cv2.resize(img, (50, 50))  # Adjust the size if needed
        img = img.flatten() / 255.0  # Normalize the image and flatten
        img = np.asarray(img).reshape(1, -1)  # Convert to 2D array
        
        # Make predictions
        prediction = svm_model.predict(img)
        class_label = class_names[int(prediction)]
        print(class_label)
        if class_label not in "Normal":
            try:
                url = "https://api.thingspeak.com/update?api_key=C6TYSSCZS42F7QCI&field3=CBE&field4="+str(class_label)
                urllib.request.urlopen(url)
                messagebox.showinfo("Alert", " HTTP request sent.")
            except Exception as e:
                messagebox.showerror("Error", f"Error sending HTTP request: {str(e)}")
            
        
        # Display the result
        result_label.config(text=f'Result: {class_label}')

# Create a tkinter window
root = tk.Tk()
root.title("Classifier")

# Create a label for the title
title_label = tk.Label(root, text="Classification", font=("Helvetica", 20))
title_label.pack(pady=20)

# Create a label to display the result
result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=20)

# Create a button to select an image
classify_button = tk.Button(root, text="Select Image", command=classify_image)
classify_button.pack()

# Create a quit button
quit_button = tk.Button(root, text="Quit", command=root.destroy)
quit_button.pack()

# Start the tkinter main loop
root.mainloop()
