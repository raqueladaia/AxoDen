# Import python libraries
import os
import sys
import tkinter as tk
from tkinter import filedialog

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from axoden.volume_projections import (
    load_table,
    process_folder,
    write_signal_intensity_along_axis_plot,
    write_summary_data_plot,
)


# Function to get the pixel size entered by the user
def get_pixel_size():
    pixel_size = float(pixel_size_entry.get())
    return pixel_size


def update_status(hndl, txt):
    hndl.config(text=txt)
    hndl.update()


def ask_folder():
    folder_path = filedialog.askdirectory()
    folder_path_entry.delete(0, tk.END)
    folder_path_entry.insert(tk.END, folder_path)

    if output_folder_path_entry.get() == "":
        output_folder_path_entry.delete(0, tk.END)
        output_folder_path_entry.insert(
            tk.END, os.path.join(folder_path, "AxoDen_output")
        )


def open_summary_csv_files(folderpath, statuslabel, endswith):
    # Check if there is a csv file in the folder path
    csv_files = [file for file in os.listdir(folderpath) if file.endswith(endswith)]
    if len(csv_files) == 0:
        update_status(
            statuslabel, "\n\nWarning:\nNo csv file found in the folder path."
        )
        return
    elif len(csv_files) > 1:
        update_status(
            statuslabel, "\n\nWarning:\nMultiple csv files found in the folder path."
        )
        return
    else:
        # Read the csv file and plot the data
        csv_file_path = os.path.join(folderpath, csv_files[0])
        table_data = load_table(csv_file_path)

    return table_data


def run_volume_projections():
    """Run the volume projections with the provided folder path and options."""

    # Get the folder path and options
    input_folder_path = folder_path_entry.get()
    output_folder_path = output_folder_path_entry.get()
    data_collection = data_collection_var.get()
    summary_data = summary_data_var.get()
    summary_axes = summary_axes_var.get()
    is_masked = is_masked_var.get()
    pixel_size = get_pixel_size()

    # Check if folder_path is empty
    if not input_folder_path:
        update_status(status_label, "\n\nWarning:\nFolder path is empty or wrong.")
        return

    if output_folder_path == "":
        output_folder_path = input_folder_path

    # Disable the run button
    run_button.config(state=tk.DISABLED)

    # Check if ONLY data collection checkbox is toggled
    if data_collection and not summary_data and not summary_axes:
        update_status(status_label, "\n\nCollecting data...")
        _, _ = process_folder(
            input_folder_path, pixel_size, is_masked, output_folder=output_folder_path
        )
        update_status(
            status_label,
            "\n\nFinished collecting and saving data.\nCheck the control plots!",
        )

    # Check if ONLY summary data checkbox is toggled
    elif summary_data and not data_collection and not summary_axes:
        # Collect and read summary csv file
        table_data = open_summary_csv_files(
            output_folder_path, status_label, "quantification.csv"
        )
        # Plot the data using the table
        update_status(status_label, "\n\nPlotting Summary Data...")
        write_summary_data_plot(output_folder_path, table_data)
        update_status(status_label, "\n\nDone")

    # Check if ONLY summary axis checkbox is toggled
    elif summary_axes and not data_collection and not summary_data:
        # Collect and read summary csv file
        table_data = open_summary_csv_files(
            output_folder_path, status_label, "axis.csv"
        )
        # Plot the data using the table
        update_status(status_label, "\n\nPlotting Summary Axis Data...")
        write_signal_intensity_along_axis_plot(
            output_folder_path, table_data, pixel_size
        )
        update_status(status_label, "\n\nDone")

    # Check if summary data and summary axes checkboxes are toggled
    elif summary_data and summary_axes and not data_collection:
        # Collect and read summary csv files
        table_data = open_summary_csv_files(
            output_folder_path, status_label, "quantification.csv"
        )
        table_data_axes = open_summary_csv_files(
            output_folder_path, status_label, "axis.csv"
        )
        # Plot the data using the tables
        update_status(status_label, "\n\nPlotting Summary Data...")
        write_summary_data_plot(output_folder_path, table_data)
        update_status(
            status_label, "\n\nPlotted Summary Data\nPlotting Summary Axis Data..."
        )
        write_signal_intensity_along_axis_plot(
            output_folder_path, table_data_axes, pixel_size
        )
        update_status(status_label, "\n\nDone")

    # Check if both data collection and summary data checkboxes are toggled
    elif summary_data and data_collection and not summary_axes:
        update_status(status_label, "\n\nCollecting data...")
        table_data, _ = process_folder(
            input_folder_path, pixel_size, is_masked, output_folder=output_folder_path
        )
        update_status(status_label, "\n\nPlotting Summary Data...")
        write_summary_data_plot(output_folder_path, table_data)
        update_status(status_label, "\n\nDone")

    # Check if summary axes checkbox and summary data checkboxes are toggled
    elif summary_axes and data_collection and not summary_data:
        update_status(status_label, "\n\nCollecting data...")
        _, table_data = process_folder(
            input_folder_path, pixel_size, is_masked, output_folder=output_folder_path
        )
        update_status(status_label, "\n\nPlotting Summary Axis Data...")
        write_signal_intensity_along_axis_plot(
            output_folder_path, table_data, pixel_size
        )
        update_status(status_label, "\n\nDone")

    # Check if ALL checkboxes are toggled
    elif data_collection and summary_axes and summary_data:
        update_status(status_label, "\n\nCollecting data...")
        table_data, table_data_axes = process_folder(
            input_folder_path, pixel_size, is_masked, output_folder=output_folder_path
        )
        update_status(status_label, "\n\nPlotting Summary Data...")
        write_summary_data_plot(output_folder_path, table_data)
        update_status(
            status_label, "\n\nPlotted Summary Data\nPlotting Summary Axis Data..."
        )
        write_signal_intensity_along_axis_plot(
            output_folder_path, table_data_axes, pixel_size
        )
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
# window.geometry("300x700")  # Set the width and height of the window
# window.minsize(900, 1600)  # Set the width and height of the window

# Create the pixel size input
pixel_size_label = tk.Label(window, text="\n\nPixel Size (um):")
pixel_size_label.pack()
pixel_size_entry = tk.Entry(window)
pixel_size_entry.insert(tk.END, "0.75521")  # Set the default pixel size
pixel_size_entry.pack()

# Create note for user
note_label = tk.Label(
    window,
    text=(
        "\nNote:\nThe default pixel size is for the 20x Objective.\n"
        "Please change it according to the objective used.\n\nx4 Objective: 3.77396\n"
    ),
)
note_label.pack()

# Call the get_pixel_size function to retrieve the pixel size entered by the user
pixel_size = get_pixel_size()

# Create the input folder path input
folder_path_label = tk.Label(window, text="Input Folder:")
folder_path_label.pack()
folder_path_entry = tk.Entry(window, width=40)
folder_path_entry.pack(padx=20)

# Create the folder path input
ask_folder_button = tk.Button(window, text="Select Folder", command=ask_folder)
ask_folder_button.pack(pady=20)

# Create the output folder path input
output_folder_path_label = tk.Label(window, text="Output Folder:")
output_folder_path_label.pack()
output_folder_path_entry = tk.Entry(window, width=40)
output_folder_path_entry.pack()

# Create the masked option
is_masked_label = tk.Label(
    window,
    text=(
        "\nCheck the box\nif images are masked and have\n"
        "values of 0 where there is no signal:"
    ),
)
is_masked_label.pack()
is_masked_var = tk.BooleanVar(value=True)
is_masked_checkbox = tk.Checkbutton(window, text="", variable=is_masked_var)
is_masked_checkbox.pack()

# Guide the user to select the options
instructions_label = tk.Label(
    window, text="\nCheck the following boxes\naccording to what you want to do:"
)
instructions_label.pack()

# Create the data collection option
data_collection_var = tk.BooleanVar(value=True)
data_collection_checkbox = tk.Checkbutton(
    window, text="Data collection", variable=data_collection_var
)
data_collection_checkbox.pack()

# Create the summary data option
summary_data_var = tk.BooleanVar(value=True)
summary_data_var_checkbox = tk.Checkbutton(
    window, text=" Plot Summary Data", variable=summary_data_var
)
summary_data_var_checkbox.pack()

# Create the summary axes option
summary_axes_var = tk.BooleanVar(value=True)
summary_axes_var_checkbox = tk.Checkbutton(
    window, text="Plot Summary Axis Data", variable=summary_axes_var
)
summary_axes_var_checkbox.pack()

# Create some space between the checkbox and the button
space_label = tk.Label(window, text="\n")
space_label.pack()

# Create the run button
run_button = tk.Button(
    window, text="Run Volume Projections", command=run_volume_projections
)
run_button.pack()

# Create label to update the status of the program
status_label = tk.Label(window, text="\n")
status_label.pack()

# Create the close button
close_button = tk.Button(window, text="Close", command=close_window)
# close_button.place(relx=0.5, rely=1.0, anchor=tk.S, y=-20)
close_button.pack(pady=20)

# Start the GUI event loop
window.mainloop()
