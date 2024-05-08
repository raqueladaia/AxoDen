# This file is used to run the streamlit app using
# python -m axoden.gui.streamlit_app

import os

import streamlit.web.bootstrap
from streamlit import config as _config

print("main from axoden.gui.streamlit_app")

dirname = os.path.dirname(__file__)
# filename = os.path.join(dirname, "axoden/gui/streamlit_app/1_📊️_Axoden.py
filename = os.path.join(dirname, "1_📊️_Axoden.py")

_config.set_option("server.headless", True)
args = []

streamlit.web.bootstrap.run(filename, args=[], flag_options={}, is_hello=False)
