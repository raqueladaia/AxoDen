"""
The `axoden` module provides functionality for performing various operations related to quantification of axonal projections.

This module contains functions that can be used to quantify axons from images.

Usage:
    import axoden

    # Initiate GUI - work in progress
    my_axoden = axoden.initiate_gui()

"""

__version__ = "0.2.0"    

from .volume_projections import process_folder, process_image
from .volume_projections import plot_summary_data, plot_signal_intensity_along_axis
from .volume_projections import write_summary_data_plot, write_signal_intensity_along_axis_plot
from .volume_projections import save_table, load_table