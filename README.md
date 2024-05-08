# Axoden
![Tests](https://github.com/raqueladaia/AxoDen/actions/workflows/test.yml/badge.svg?branch=main)
![Linting](https://github.com/raqueladaia/AxoDen/actions/workflows/lint.yml/badge.svg?branch=main)


## TODOs

- [ ] add one figure from the paper here that describes what AxoDen does
- [ ] set up github token to make it possible to upload release directly from gh action
- [ ] Raquel to write quick overview here in README.md
- [ ] Test deployed streamlit app with two concurrent users and 100+ files each
- [ ] Decide on upload limit for deployed app
- [ ] Add citation here once publication is submitted
- [ ] Add citation to streamlit app once publication is submitted
- [ ] Add link to publication
- [ ] make repo public once we're ready :muscle:

- [ ] Raquel to write text for initial tutorial --> Text needs to be improved.
- [ ] Raquel to write section about how data is used on streamlit tutorial page. See readme.
- [ ] Raquel to go over main streamlit app and correct/adapt naming
- [ ] Raquel to double check that the units are correct, volume_projections.py line 443 seems strange
- [x] Raquel to include sample images

- [x] Pascal to set up gh action for unit testing
- [x] Pascal to set up gh action for linting
- [x] Pascal to set up gh action to build standalone executables
- [ ] Pascal to add badge for pypi
- [ ] Pascal to create gh action to update pip module from release branch

## Table of Contents

- [How to use AxoDen](#how-to-use-axoden)
  - [Streamlit App](#streamlit-app)
  - [Standalone GUI](#standalone-gui)
- [Installation](#installation)
  - [With pip](#pip-install)
  - [Git clone](#clone-from-github)
- [Contributing](#contributing)

## How To Use AxoDen

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
- The filename of the images matter! Rename the files to the following format: _subjectID_brainregion_group_WhateverYouWant.tif_
> [!TIP]
>  - Do not use spaces in the name of the file
>  - The underscore ("_") character is used to separate the _subjectID_, _brainregion_ and _group_
>  - At least _subjectID_ and _brainregion_ are required, e.g. "animal1_ACC.tif" is valid, "myfile.tif" does not work
>  - The _group_ is optional, but can be helpful if you want to further process the output from Axoden, e.g. if you want to do your own statistics.

#### Image Masking

If you want to mask your images to analyze a precise brain region:
1. Use Adobe Illustrator or Power Point to overlay the brain atlas onto your image and set the boundaries of your brain region of interest.
2. Download ImageJ or FIJI if you do not have it installed.
3. Open the image of interest.
4. Select the freehand selection tool on the top menu.
5. Draw the boundaries of the region of interest according to the overlay you created in step 1. On the top menu select in order:
    - `Edit > Clear Outside`
    - `Image > Crop`
6. If the signal to noise ratio needs to be enhanced:
    - On the top menu select `Image > Adjust > Brightness > Crop`
    - Decrease the "Maximum" value using the sliding bar to increase fluorescence intensity.
    - Increase the "Minimum" value using the sliding bar to decrease background fluorescence
    - Tip: Use the histogram (`Analyze > Histogram`) to confirm that the background fluorescence in the brain region does not contain zero values.
7. Save the resulting image following the [AxoDen naming conventions](#naming-convention).

#### Running the GUI
<img src="https://github.com/raqueladaia/AxoDen/blob/main/media/tkinter_gui.png?raw=true)" width="300">

- Select the folder you prepared in the previous step. Use the `Select Folder` button to open a folder selection dialog
- The output folder will be set to the subfolder `Axoden_output` but you can change it. If the folder does not exist, it will be created.
- Make sure the pixel size is set to the correct value.
This depends on both the objective used and your device!
- If you use masked images as described in step [Image Masking](#image-masking), leave the check as it is. If you did not mask the images and want to process original, unmasked images, remove the check.
- By default, AxoDen will collect all data, then plot the summary and finally the summary axis data.
- Finally click `Run Volume Projections` to start the processing.
- You will find the results in the _Output Folder_.

## Installation

> [!Note]
> This section applies only to people who want to use AxoDen in their python code, or want to run AxoDen from code.

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

If you get an error saying TkInter is not installed, see the [Install TkInter](#install-tkinter) section.

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

We suggest to set up a virual environment. For more information regarding virtual environments, see https://docs.python.org/3/library/venv.html

#### Install the dependencies

Use pip to install the dependencies (typically in your virtual environment):

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

#### Running the GUI

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
usually at http://localhost:8501.


This is the same interface as the [Streamlit Web App](https://axoden.streamlit.app).

# Using axoden in your code
Below you find a few examples of how to use axoden.

You can process a single image. This will not save any results or plots.
```python
import axoden

img_path = "test_images/745_TH-CL.tif"
fig, data, data_axis = axoden.process_image(img_path, is_masked=True, pixel_size=0.75521)
fig.show()
```

Or you can process all images in a folder and save the results, similar to what the gui does:
```python
import axoden

input_folder = "test_images"
output_folder = "results"
df_data, df_data_axis = axoden.process_folder(input_folder, pixel_size=0.75521, is_masked=True, output_folder=output_folder)
axoden.write_summary_data_plot(output_folder, df_data, project_name="my project")
axoden.write_signal_intensity_along_axis_plot(output_folder, df_data_axis, pixel_size=0.75521, project_name="my_project")
```

It is also possible to not save any output, this might be useful if you further process the results.
```python
import axoden

input_folder = "test_images"
df_data, df_data_axis = axoden.process_folder(input_folder, pixel_size=0.75521, is_masked=True, save=False)

output_folder = "results"
axoden.save_table(df_data, output_folder, "projections_quantifications.csv")
axoden.save_table(df_data_axis, output_folder, "projections_quantification_along_axis.csv")

fig_data = axoden.plot_summary_data(df_data, "my_project")
fig_data_axis = axoden.plot_signal_intensity_along_axis("my_project", df_data_axis, pixel_size=0.75521)

fig_data.show()
fig_data_axis.show()
```
