"""
The `axoden` module provides functionality for performing various operations related to
quantification of axonal projections.

This module contains functions that can be used to quantify axons from images.

Usage:
    import axoden

    # Initiate GUI - work in progress
    my_axoden = axoden.initiate_gui()

"""

__version__ = "0.2.0"

from .volume_projections import (  # noqa: F401
    load_table,
    plot_signal_intensity_along_axis,
    plot_summary_data,
    process_folder,
    process_image,
    save_table,
    write_signal_intensity_along_axis_plot,
    write_summary_data_plot,
)
