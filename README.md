# TMA4220 Project 1  [![Build Status](https://travis-ci.com/Jimpachnet/TMA4220_project_1.svg?token=oWG7xeW32J8widyCtTw8&branch=master)](https://travis-ci.com/Jimpachnet/TMA4220_project_1)
We present our solution to project 1 in the course TMA4220 at NTNU Trondheim in autumn semester 2018.
The code impelements all required components to solve simple evolutionary and static PDEs in 2D on a triangular mesh. In particular, we implemented a solver for the Helmholtz problem and the homogenous unsteady heat equation. A third solver extends the dynamic solver to cope with the homogenous wave equoation in 2D.
![Solution to the Helmholtz problem](https://i.imgur.com/HrLQhM3.png)
## Installation
The program is written in Python 3. The package is installed via the 
### Requirements
Additional requirements are listed in the setup.py and requirements.txt, namely
* scipy
* matplotlib
* numpy
* tqdm
With '''pip''' these packages can be installed by using
'''$ pip install -r requirements.txt'''
## Execution
The program is started by executing the main file located in the root. Without any arguments passed, the software will conduct an error analysis for the Helmholtz problem.
### Arguments
* -v Visualizes the analytical solution of the Helmholtz problem for debugging the basic functionality of the vis functions.
* -vd Visualizes the analytical solution of the time dependant problem for debugging.
* -m Generates and plots a triangular mesh for debugging.
* -s Solves the Helmholtz problem and plots the solution
* -sd Solves the dynamic problem and generates an animation of the solution
* -w Solves the 2D wave equation and generates an animation 
## State
### Features
* f(x) implemented
* Analytical u(x) implemented
* Visualization of the 2D functions
* Simple unit tests for the functions
### Todo
* Find axis 3d inversion bug
* Check if boundary conditions are implemented right
* Intgrating for b - how to incorporate determinant
