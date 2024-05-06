# Project Name

A brief description of the project.

## Installation

### Git clone
#### Clone the repo with git
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
pip install -r requirements.txt
```

This will install all required python dependencies, except the tkinter libraries, which has to be installed separately.

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

### Install via pip


## Usage

### Streamlit Locally

Assuming you have cloned the repo using the steps above and installed the dependencies, you can run the streamlit app locally using your virtual python environment.
Make sure to have your crea
```bash
  streamlit run axoden/streamlit_app/1_📊️_Axoden.py
```
This is the same interface that is running as the [Streamlit Web App](axoden.streamlit.app), but when running it locally you will benefit from faster processing time, and uploads should be almost instantaneous.


## Table of Contents

- [Installation](#installation)
  - via pip
  - git clone
  - streamlit
- [Usage](#usage)
  - run the gui
  - use python module
  - run streamlit
- [Contributing](#contributing)
- [License](#license)
  - add license file

## Installation

Instructions on how to install and set up the project.

## Usage

Instructions on how to use the project and any relevant examples.

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