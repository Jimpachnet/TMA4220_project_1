# TMA4220 Project 1  [![Build Status](https://travis-ci.com/Jimpachnet/TMA4220_project_1.svg?token=oWG7xeW32J8widyCtTw8&branch=master)](https://travis-ci.com/Jimpachnet/TMA4220_project_1)
We present our solution to project 1 in the course TMA4220 at NTNU Trondheim in autumn semester 2018.
The code impelements all required components to solve simple evolutionary and static PDEs in 2D on a triangular mesh. In particular, we implemented a solver for the Helmholtz problem and the homogenous unsteady heat equation. A third solver extends the dynamic solver to cope with the homogenous wave equoation in 2D.
![Solution to the Helmholtz problem](https://i.imgur.com/HrLQhM3.png)
## Installation
The program is written in Python 3. The installation is shown via ```pip```.
### Requirements
Additional requirements are listed in the setup.py and requirements.txt, namely
* scipy
* matplotlib
* numpy
* tqdm

With ```pip``` these packages can be installed by using ```$ pip install -r requirements.txt```. We recommend to create a new virtual environment using ```venv```. Afterwards, install the project using pip: ```$ pip install -e /path/to/parent/folder/of/setup.py```. Now, the main file of the project can be executed by navigating into ```project_1``` and executing ```$ python main.py```. 
## Execution
The program is started by executing the main file located in the root. Without any arguments passed, the software will calculate and plot the solution to the Helmholtz problem.
### Arguments
* ```-v``` Visualizes the analytical solution of the Helmholtz problem for debugging the basic functionality of the vis functions.
* ```-vd``` Visualizes the analytical solution of the time dependant problem for debugging.
* ```-m``` Generates and plots a triangular mesh for debugging.
* ```-s``` Solves the Helmholtz problem and plots the solution
* ```-sd``` Solves the unsteady heat equation and generates an animation of the solution as well as a picture of the final frame. [Video of the solution](https://goo.gl/qCdw5e)
* ```-w``` Solves the 2D wave equation and generates an animation 
* ```-r``` Genaterates the required plots for the report, including the error analysis
