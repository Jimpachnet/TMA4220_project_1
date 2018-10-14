[![Build Status](https://travis-ci.com/Jimpachnet/TMA4220_project_1.svg?token=oWG7xeW32J8widyCtTw8&branch=master)](https://travis-ci.com/Jimpachnet/TMA4220_project_1)
# TMA4220 Project 1
Solution to project 1 in the course TMA4220 at NTNU Trondheim in autumn semester 2018.
![Solution to the Helmholtz problem](https://image.ibb.co/eOHEJp/helmholtz.png)
## Documentation
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
