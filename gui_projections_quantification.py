import os
import pandas as pd
import tkinter as tk

from tkinter import ttk
from volume_projections import collect_data, plot_data


# Function to get the pixel size entered by the user
def get_pixel_size():
    pixel_size = float(pixel_size_entry.get())
    return pixel_size

def update_status(hndl, txt):
    hndl.config(text=txt)
    hndl.update()

# Function to run the volume_projections.py file with the provided folder path and options
def run_volume_projections():

    # Get the folder path and options
    folder_path = folder_path_entry.get()
    data_collection = data_collection_var.get()
    data_analysis = data_analysis_var.get()
    pixel_size = get_pixel_size()

    # Check if folder_path is empty
    if not folder_path:
        update_status(status_label, "\n\nWarning:\nFolder path is empty or wrong.")
        return

    # Disable the run button
    run_button.config(state=tk.DISABLED)

    # Check if data collection checkbox is toggled
    if data_collection and not data_analysis:
        update_status(status_label, "\n\nCollecting data...")
        _ = collect_data(folder_path, pixel_size)
        update_status(status_label, "\n\nFinished collecting and saving data!")
    
    # Check if data analysis checkbox is toggled
    elif data_analysis and not data_collection:
        # Check if there is a csv file in the folder path
        csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]
        if len(csv_files) == 0:
            update_status(status_label, "\n\nWarning:\nNo csv file found in the folder path.")
            return
        elif len(csv_files) > 1:
            update_status(status_label, "\n\nWarning:\nMultiple csv files found in the folder path.")
            return
        else:
            update_status(status_label, "\n\nData analysis and plotting...")
            # Read the csv file and plot the data
            csv_file_path = os.path.join(folder_path, csv_files[0])
            table_data = pd.read_csv(csv_file_path)            
            # Plot the data using the table
            plot_data(folder_path, table_data)
            update_status(status_label, "\n\nDone")

    # Check if both data collection and data analysis checkboxes are toggled
    elif data_analysis and data_collection:
        update_status(status_label, "\n\nCollecting data...")
        table_data = collect_data(folder_path, pixel_size)
        update_status(status_label, "\n\nData analysis and plotting...")
        plot_data(folder_path, table_data)
        update_status(status_label, "\n\nDone")

    # Enable the run button
    run_button.config(state=tk.NORMAL)



# Function to close the window
def close_window():
    window.destroy()



# Create the main window
window = tk.Tk()
window.title("Quantification of Projections")

# Configure the window size
window.geometry("300x400")  # Set the width and height of the window

# Create the pixel size input
pixel_size_label = tk.Label(window, text="\n\nPixel Size (um):")
pixel_size_label.pack()
pixel_size_entry = tk.Entry(window)
pixel_size_entry.insert(tk.END, "0.75521")  # Set the default pixel size
pixel_size_entry.pack()

# Create note for user
note_label = tk.Label(window, text="\nNote:\nThe default pixel size is for the 20x Objective.\nPlease change it according to the objective used.\n")
note_label.pack()

# Call the get_pixel_size function to retrieve the pixel size entered by the user
pixel_size = get_pixel_size()

# Create the folder path input
folder_path_label = tk.Label(window, text="Folder Path:")
folder_path_label.pack()
folder_path_entry = tk.Entry(window)
folder_path_entry.pack()

# Create the data collection option
data_collection_var = tk.BooleanVar()
data_collection_checkbox = tk.Checkbutton(window, text="Data collection", variable=data_collection_var)
data_collection_checkbox.pack()

# Create the data analysis option
data_analysis_var = tk.BooleanVar()
data_analysis_checkbox = tk.Checkbutton(window, text="Data analysis and plotting", variable=data_analysis_var)
data_analysis_checkbox.pack()

# Create the run button
run_button = tk.Button(window, text="Run Volume Projections", command=run_volume_projections)
run_button.pack()

# Create label to update the status of the program    
status_label = tk.Label(window, text="\n\n")
status_label.pack()  

# Create the close button
close_button = tk.Button(window, text="Close", command=close_window)
close_button.pack(side=tk.BOTTOM)

# Start the GUI event loop
window.mainloop()
