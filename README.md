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

# Done
- themeing, colors etc. -> UPenn colors ok

# TODO
- Raquel
  - text for initial tutorial
  - initial citation -> waiting for submission
  - go over all headers, text, naming
  - test how many images can be uploaded/used simultaneously --> Todo later with real images
    - how should we limit file sizes?
    - limit number of images
  - how many concurrent users? How can we test?
- Pascal
  - what to do if image names are in the wrong format? -> detect, warn and abort --> Pascal to increase robustness
  - better error handling --> Pascal
  - deploy from raqueladaia/AxoDen --> organize call 1-2h
  - use release branch for this
- how is the axoden pip package used? --> Pascal to propose
  - there should be a few simple examples: python -m axoden_gui? python -m axoden
- what sample images to include
