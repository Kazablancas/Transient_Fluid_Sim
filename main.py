import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt

# Set appearance mode: 'System', 'Dark', or 'Light'
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")  # You can try "dark-blue", "green", etc.

# Create the main app window
app = ctk.CTk()
app.geometry("1500x900")
app.title("Airfoil Simulation GUI")

# Add a label
label = ctk.CTkLabel(app, text="Airfoil Simulator", font=("Arial", 24))
label.pack(pady=20)

# Button callback function
def import_airfoil():
    print("Import Airfoil button clicked!")

# Add a button
button = ctk.CTkButton(app, text="Import Airfoil", command=import_airfoil)
button.pack(pady=10)

# Start the GUI loop
app.mainloop()

