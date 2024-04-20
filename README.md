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
