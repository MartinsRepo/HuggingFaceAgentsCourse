# Installation in Ubuntu
## 1. Prerequisites
**(a) Python Version used:** 3.12.9

**(b) Activate CONDA Environment**

A virtual environment is a Python environment such that the Python interpreter, libraries and scripts installed into it are isolated from those installed in other virtual environments. Here **conda** is used.
We will generate different **venv**s for every subproject.

* Subprojects stored in the folder **notebooks** are:
    + smolAgents
    + llamaIndexAgents

> cd notebooks/<subproject>

> conda create --name <subproject_name> python=3.12.9

> conda activate <subproject_name>

eg. conda activate llamaIndexAgents

## 2. Installing the Requirements

### (a) SmolAgents

> cd notebooks/smolAgents/

> pip install -r requirements.txt

### (b) LlamaIndex

> cd notebooks/llamaIndexAgents/

> pip install -r requirements.txt

> Written with [StackEdit](https://stackedit.io/). 
