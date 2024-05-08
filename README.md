# Axoden
![Tests](https://github.com/raqueladaia/AxoDen/actions/workflows/test.yml/badge.svg?branch=main)
![Linting](https://github.com/raqueladaia/AxoDen/actions/workflows/lint.yml/badge.svg?branch=main)
![License](https://github.com/raqueladaia/AxoDen/blob/main/LICENSE)

TODO: add badges for pypi, license

TODO: Raquel to write quick overview

## Table of Contents

- [Usage](#usage)
- [Installation](#installation)
  - [With pip](#pip-install)
  - [Git clone](#clone-from-github)
- [Contributing](#contributing)
- [License](#license)
  - add license file

## Usage

The simplest way to use AxoDen is to simply use the deployed web app at [axoden.streamlit.app](https://axoden.streamlit.app). It will allow you to upload images directly through a browser and download the results.
The files that you upload are only ever kept in memory and discarded once you disconnect from the web app. And we never store or collect anything you upload.

However, if you do not want to upload the images or need to process large amounts of images,
we provide a standalone application with a GUI that you can [download from here](https://github.com/raqueladaia/AxoDen/releases).

### Streamlit App

Whether you use the deployed app [axoden.streamlit.app](https://axoden.streamlit.app) or run it locally (see [Installation](#installation)), the easiest way to get started is to check the **Tutorial & How To** page that you see on the left pane when the app starts.
This should explain the steps required and also give you a few tips on how to prepare the images before you upload them.

### Standalone GUI

If you download the standalone gui from our [release page](https://github.com/raqueladaia/AxoDen/releases), there are a few considerations:

#### Preparing folder and images
Axoden processes **folders**, not single files.
So all the files that you want to process should be in a folder you prepare.

#### Naming Convention
- All images need to be in **.tif format**.
- The filename of the images matter.
  - Rename the files to the following format: _subjectID_brainregion_group_WhateverYouWant.tif_
  - Tip: Do not use spaces in the name of the file
  - Tip: the underscore ("_") character is used to separate the _subjectID_, _brainregion_ and _group_
  - Tip: at least _subjectID_ and _brainregion_ are required, e.g. "animal1_ACC.tif" is valid, "myfile.tif" does not work
  - Tip: The _group_ is optional, but can be helpful if you want to further process the output from Axoden, e.g. if you want to do your own statistics.

#### Image Masking

If you want to mask your images to analyze a precise brain region:
1. Use Adobe Illustrator or Power Point to overlay the brain atlas onto your image and set the boundaries of your brain region of interest.
2. Download ImageJ or FIJI if you do not have it installed.
3. Open the image of interest.
4. Select the freehand selection tool on the top menu.
5. Draw the boundaries of the region of interest according to the overlay you created in step 1.
  On the top menu select in order:
  - `Edit > Clear Outside`
  - `Image > Crop`
6. If the signal to noise ratio needs to be enhanced:
  - On the top menu select `Image > Adjust > Brightness > Crop`
  - Decrease the "Maximum" value using the sliding bar to increase fluorescence intensity.
  - Increase the "Minimum" value using the sliding bar to decrease background fluorescence
  - Tip: Use the histogram (`Analyze > Histogram`) to confirm that the background fluorescence in the brain region does not contain zero values.
7 Save the resulting image following the [AxoDen naming conventions](#naming-convention).

#### Running the GUI
![TkInter GUI](https://github.com/raqueladaia/AxoDen/blob/main/media/tkinter_gui.png?raw=true)

## Installation

There are two ways to install axoden: Through pip (recommended) or by cloning the repo from github.

### Pip Install

For most users, this is the preferred option. Install axoden through pip:
```bash
  pip install axoden
```

If you also want the extra gui dependencies, specify it in the pip install:
```bash
  pip install axoden[gui]
```

#### Running the GUIs

Assuming you installed the gui dependencies with the above `pip install axoden[gui]`,
you can start the tkinter with
```bash
  python -m axoden.gui.gui_projections_quantification
```

To start the streamlit app locally, run:
```bash
  python -m axoden.gui.streamlit_app
```

This is the same interface that is running as the [Streamlit Web App](https://axoden.streamlit.app), but when running it locally you will benefit from faster processing time, and uploads should be almost instantaneous.


### Clone from github

If you want to make changes to axoden, you can clone the full code from the github repo.

#### Clone the repo
```bash
  git clone https://github.com/raqueladaia/AxoDen.git
```

#### Create a virtual environment
```bash
  python3.11 -m venv .venv
```

#### Activate the created virtual environment.

Linux, Mac:
```bash
  source .venv/bin/activate
```

Window:
```bash
  .\.venv\Scripts\activate.bat
```

For more information regarding virtual environments, see https://docs.python.org/3/library/venv.html

#### Install the dependencies

With your virtual environment activated, use pip to install the dependencies:

```bash
pip install -r requirements_gui.txt
```

This will install all required python dependencies to fully use axoden, including running the streamlit app locally. The tkinter library cannot be installed through pip and has to be installed separately.

There are two other requirements files that you might want to use instad of the above.
- `requirements.txt`: Excludes streamlit, but allows you to use axoden as a library, as well as use the tkinter gui.
- `requirements_dev.txt`: For development purposes, includes everything including testing and linting (tox, pytest, black, ruff).

#### Install TkInter

For windows, tkinter is installed when installing python.
If you get an error complaining the module named 'tkinter' is not found when executing the python code, you will have to modify the python installation. For that, launch the python installation again and in the installer, select 'Modify'. Make sure that the checkbox 'tcl/tk and IDLE' is checked!

For linux:
```bash
sudo apt install python3-tk
```

Or for a specific version of python, e.g. python 3.11:
```bash
sudo apt install python3.11-tk
```

Mac (not tested):
```bash
brew install python-tk
```

#### Running Tkinter

Simply execute the file `axoden/gui/gui_projections_quantification.py`, e.g. through a terminal:
```bash
  python axoden/gui/gui_projections_quantification.py
```

#### Running Streamlit App Locally

You can run the streamlit app locally with

```bash
  streamlit run axoden/gui/streamlit_app/1_📊️_Axoden.py
```

This should automatically open a browser window to the app.
If not, streamlit will post the URL where you can reach the web app,
usually at (http://localhost:8501)


This is the same interface as the [Streamlit Web App](https://axoden.streamlit.app).

## Contributing

Guidelines for contributing to the project and how to submit changes.

## License

Information about the project's license and any relevant terms.

# Done Pascal
- deploy from raqueladaia/AxoDen --> organize call 1-2h
- use release branch for this
- how is the axoden pip package used? --> Pascal to propose
- themeing, colors etc. -> UPenn colors ok
- bug fixes, see below
- csv loading and saving is compatible with saving csv data from streamlit (same format)
- proposal for axoden use as a package
- docstrings for common functions
- gui fixes and improvements, see below

# TODO
- Raquel
  - text for initial tutorial --> Text needs to be improved.
  - double check that the units are correct, volume_projections.py line 377 seems strange
  - initial citation -> waiting for submission
  - go over all headers, text, naming
  - test how many images can be uploaded/used simultaneously --> Todo later with real images
    - how should we limit file sizes?
    - limit number of images
  - how many concurrent users? How can we test?
  - what sample images to include
  - write section about how data is used on streamlit tutorial page: data is kept in memory
- Pascal
  - write section about how data is used in README.md: data is kept in memory
  - remove guis from pip module
  - pip module needs dependencies
  - github action to build .exe for windows users
  - update README with instructions for
    - streamlit use locally
    - gui locally
    - pip install
    - how to use axoden functionality


# Proposal axoden use
```
# process a single image
img_path = "samples_images/0001_dsRed.tif"
fig, data, data_axis = axoden.process_image(file_name, is_masked=True, pixel_size=0.75)
fig.show()

# process a folder and save the results
input_folder = "samples/"
output_folder = "results"
df_data, df_data_axis = axoden.process_folder(input_folder, pixel_size=0.75, is_masked=True, output_folder=output_folder)
axoden.write_summary_data_plot(output_folder, df_data, project_name = "my project"):
axoden.write_signal_intensity_along_axis_plot(output_folder, df_data_axis, pixel_size=0.75, project_name = "my_project")


# process a folder and show the results
input_folder = "samples/"
df_data, df_data_axis = axoden.process_folder(input_folder, pixel_size=0.75, is_masked=True, save=False)

output_folder = "results"
axoden.save_table(df_data, output_folder, "projections_quantifications.csv")
axoden.save_table(df_data_axis, output_folder, "projections_quantification_along_axis.csv")

fig_data = axoden.plot_summary_data(df_data, "my_project")
fig_data_axis = axoden.plot_signal_intensity_along_axis("my_project", df_data_axis, pixel_size=0.75):

fig_data.show()
fig_data_axis.show()
```

# Changes for tk GUI
- added folder dialog
- set boolean defaults
- removed window.geometry(...), this was too small, removing it sets it to fit everything in the window automatically. But it might behave differently on windows, need to check!
- add output folder --> separates input images and output 
- suggestion: remove separate collect and plot data/axis to make it simpler? But it's working right now

# Bugs fixed:
## saving axis data is saving it as the string representation, not the full data!
## loading animal col from csv resuled in int colum, so animal named "0004" would become 4

animal,brain_area,signal_bin_x_ax,signal_bin_y_ax,signal_gray_x_ax,signal_gray_y_ax
0004,Insula-GFP,[0 0 0 ... 3 3 1],[0 0 0 ... 0 0 0],[322. 335. 351. ... 505. 547. 461.],[ 39.  83. 134. ... 123.  76.  23.]
0006,NAcc,[40 40 47 ... 13 13 14],[  0   0   0 ... 270 264 250],[4769. 4796. 4871. ... 4907. 5176. 5235.],[7.0000e+00 1.6000e+01 3.0000e+01 ... 1.7350e+04 1.7080e+04 1.6103e+04]
0005,Insula-GFP,[0 0 0 ... 3 3 1],[0 0 0 ... 0 0 0],[322. 335. 351. ... 505. 547. 461.],[ 39.  83. 134. ... 123.  76.  23.]
0002,Insula-GFP,[0 0 0 ... 3 3 1],[0 0 0 ... 0 0 0],[322. 335. 351. ... 505. 547. 461.],[ 39.  83. 134. ... 123.  76.  23.]
0001,Insula-GFP,[0 0 0 ... 3 3 1],[0 0 0 ... 0 0 0],[322. 335. 351. ... 505. 547. 461.],[ 39.  83. 134. ... 123.  76.  23.]
0001,NAcc,[40 40 47 ... 13 13 14],[  0   0   0 ... 270 264 250],[4769. 4796. 4871. ... 4907. 5176. 5235.],[7.0000e+00 1.6000e+01 3.0000e+01 ... 1.7350e+04 1.7080e+04 1.6103e+04]