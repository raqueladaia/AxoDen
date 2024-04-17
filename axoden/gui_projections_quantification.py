# Import python libraries
import os
import re
import numpy as np
import pandas as pd
import tkinter as tk

# Import the functions from the volume_projections.py file
from volume_projections import collect_data, plot_summary_data, plot_signal_intensity_along_axis

# Function to get the pixel size entered by the user
def get_pixel_size():
    pixel_size = float(pixel_size_entry.get())
    return pixel_size

def update_status(hndl, txt):
    hndl.config(text=txt)
    hndl.update()

def open_summary_csv_file(folderpath, statuslabel, endswith):
    # Check if there is a csv file in the folder path
    csv_files = [file for file in os.listdir(folderpath) if file.endswith(endswith)]
    if len(csv_files) == 0:
        update_status(statuslabel, "\n\nWarning:\nNo csv file found in the folder path.")
        return
    elif len(csv_files) > 1:
        update_status(statuslabel, "\n\nWarning:\nMultiple csv files found in the folder path.")
        return
    else:
        # Read the csv file and plot the data
        csv_file_path = os.path.join(folderpath, csv_files[0])
        table_data = pd.read_csv(csv_file_path)
        if 'axis' in endswith:
            for col in table_data.columns:
                if 'signal' in col:
                    table_data[col] = table_data[col].apply(lambda x: np.array([float(n) for n in re.findall('\d+', x)]))
    return table_data
    

# Function to run the volume_projections.py file with the provided folder path and options
def run_volume_projections():

    # Get the folder path and options
    folder_path = folder_path_entry.get()
    data_collection = data_collection_var.get()
    summary_data = summary_data_var.get()
    summary_axes = summary_axes_var.get()
    shape_rectangle = rectangle_shape_var.get()
    pixel_size = get_pixel_size()

    # Check if folder_path is empty
    if not folder_path:
        update_status(status_label, "\n\nWarning:\nFolder path is empty or wrong.")
        return

    # Disable the run button
    run_button.config(state=tk.DISABLED)

    # Check if ONLY data collection checkbox is toggled
    if data_collection and not summary_data and not summary_axes:
        update_status(status_label, "\n\nCollecting data...")
        _, _ = collect_data(folder_path, pixel_size, shape_rectangle)
        update_status(status_label, "\n\nFinished collecting and saving data.\nCheck the control plots!")
        
    # Check if ONLY summary data checkbox is toggled
    elif summary_data and not data_collection and not summary_axes:
        # Collect and read summary csv file
        table_data = open_summary_csv_file(folder_path, status_label, "quantification.csv")
        # Plot the data using the table
        update_status(status_label, "\n\nPlotting Summary Data...")
        plot_summary_data(folder_path, table_data)
        update_status(status_label, "\n\nDone")

    # Check if ONLY summary axis checkbox is toggled
    elif summary_axes and not data_collection and not summary_data:
        # Collect and read summary csv file
        table_data = open_summary_csv_file(folder_path, status_label, "axis.csv")
        # Plot the data using the table
        update_status(status_label, "\n\nPlotting Summary Axis Data...")
        plot_signal_intensity_along_axis(folder_path, table_data, pixel_size)
        update_status(status_label, "\n\nDone")
    
    # Check if summary data and summary axes checkboxes are toggled
    elif summary_data and summary_axes and not data_collection:
        # Collect and read summary csv files
        table_data = open_summary_csv_file(folder_path, status_label, "quantification.csv")
        table_data_axes = open_summary_csv_file(folder_path, status_label, "axis.csv")
        # Plot the data using the tables
        update_status(status_label, "\n\nPlotting Summary Data...")
        plot_summary_data(folder_path, table_data)
        update_status(status_label, "\n\nPlotted Summary Data\nPlotting Summary Axis Data...")
        plot_signal_intensity_along_axis(folder_path, table_data_axes, pixel_size)
        update_status(status_label, "\n\nDone")

    # Check if both data collection and summary data checkboxes are toggled
    elif summary_data and data_collection and not summary_axes:
        update_status(status_label, "\n\nCollecting data...")
        table_data, _ = collect_data(folder_path, pixel_size, shape_rectangle)
        update_status(status_label, "\n\nPlotting Summary Data...")
        plot_summary_data(folder_path, table_data)
        update_status(status_label, "\n\nDone")

    # Check if summary axes checkbox and summary data checkboxes are toggled
    elif summary_axes and data_collection and not summary_data:
        update_status(status_label, "\n\nCollecting data...")
        _,  table_data = collect_data(folder_path, pixel_size, shape_rectangle)
        update_status(status_label, "\n\nPlotting Summary Axis Data...")
        plot_signal_intensity_along_axis(folder_path, table_data, pixel_size)
        update_status(status_label, "\n\nDone")

    # Check if ALL checkboxes are toggled
    elif data_collection and summary_axes and summary_data:
        update_status(status_label, "\n\nCollecting data...")
        table_data, table_data_axes = collect_data(folder_path, pixel_size, shape_rectangle)
        update_status(status_label, "\n\nPlotting Summary Data...")
        plot_summary_data(folder_path, table_data)
        update_status(status_label, "\n\nPlotted Summary Data\nPlotting Summary Axis Data...")
        plot_signal_intensity_along_axis(folder_path, table_data_axes, pixel_size)
        update_status(status_label, "\n\nDone")
    
    # Enable the run button
    run_button.config(state=tk.NORMAL)



# Function to close the window
def close_window():
    window.destroy()

# Create the main window
window = tk.Tk()
window.title("AxoDen")

# Configure the window size
window.geometry("300x700")  # Set the width and height of the window

# Create the pixel size input
pixel_size_label = tk.Label(window, text="\n\nPixel Size (um):")
pixel_size_label.pack()
pixel_size_entry = tk.Entry(window)
pixel_size_entry.insert(tk.END, "0.75521")  # Set the default pixel size
pixel_size_entry.pack()

# Create note for user
note_label = tk.Label(window, text="\nNote:\nThe default pixel size is for the 20x Objective.\nPlease change it according to the objective used.\n\nx4 Objective: 3.77396\n")
note_label.pack()

# Call the get_pixel_size function to retrieve the pixel size entered by the user
pixel_size = get_pixel_size()

# Create the folder path input
folder_path_label = tk.Label(window, text="Folder Path:")
folder_path_label.pack()
folder_path_entry = tk.Entry(window)
folder_path_entry.pack()

# Create the rectangle shape option
rectangle_label = tk.Label(window, text="\nCheck the box\nif the image has a rectangular shape:")
rectangle_label.pack()
rectangle_shape_var = tk.BooleanVar()
rectangle_shape_checkbox = tk.Checkbutton(window, text="", variable=rectangle_shape_var)
rectangle_shape_checkbox.pack()

# Guide the user to select the options
instructions_label = tk.Label(window, text="\nCheck the following boxes\naccording to what you want to do:")
instructions_label.pack()

# Create the data collection option
data_collection_var = tk.BooleanVar()
data_collection_checkbox = tk.Checkbutton(window, text="Data collection", variable=data_collection_var)
data_collection_checkbox.pack()

# Create the summary data option
summary_data_var = tk.BooleanVar()
summary_data_var_checkbox = tk.Checkbutton(window, text=" Plot Summary Data", variable=summary_data_var)
summary_data_var_checkbox.pack()

# Create the summary axes option
summary_axes_var = tk.BooleanVar()
summary_axes_var_checkbox = tk.Checkbutton(window, text="Plot Summary Axis Data", variable=summary_axes_var)
summary_axes_var_checkbox.pack()

# Create some space between the checkbox and the button
space_label = tk.Label(window, text="\n")
space_label.pack()

# Create the run button
run_button = tk.Button(window, text="Run Volume Projections", command=run_volume_projections)
run_button.pack()

# Create label to update the status of the program    
status_label = tk.Label(window, text="\n\n")
status_label.pack()  

# Create the close button
close_button = tk.Button(window, text="Close", command=close_window)
close_button.place(relx=0.5, rely=1.0, anchor=tk.S, y=-20)

# Start the GUI event loop
window.mainloop()
