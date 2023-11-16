#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt
ipython kernel install --user --name=venvPython3.9.10 --display-name="Python venv 3.9.10"
