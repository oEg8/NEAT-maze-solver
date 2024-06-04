# Maze Solver with NEAT and Visualization

This repository contains Python scripts for solving mazes using the NeuroEvolution of Augmented Topologies (NEAT) algorithm and visualizing the process.

## Contents

- [Description](#description)
- [Requirements](#requirements)
- [Usage](#usage)
- [Scripts](#scripts)
- [Contributors](#contributors)
- [License](#license)
- [References](#references)

## Description

The project includes three main Python scripts:

1. **neat_solver.py**: Implements a maze-solving algorithm using NEAT. Given a maze, the solver uses NEAT to evolve a solution, finding the optimal path from the start to the goal.
   
2. **maze_maker.py**: Generates randomized mazes with specified characteristics, such as size, obstacle density, and minimum route length. The mazes are represented as numpy arrays, with different values denoting empty cells, walls, start, and end points.

3. **visualizer.py**: Provides a visualization of a maze-solving process. It uses pygame to create a graphical interface where the NEAT solver navigates through the maze, displaying the path and progress in real-time.

4. **neat-configuration.txt**: Configuration file where all parameters are defined. 

## Requirements

- Python 3.x
- numpy
- neat-python
- pygame

You can install the dependencies using pip:

```
pip install numpy neat-python pygame
```

## Usage

- Run `NEATclass.py` This file runs everything.

## Scripts

- `neat_solver.py`: NEAT algorithm implementation for solving mazes.
- `maze_maker.py`: Maze generation script.
- `visualizer.py`: Visualization script for maze solving.
- `neat-configuration.txt`: The configuration file for NEAT

## Contributors

- https://github.com/oEg8

## License

This project is licensed under the MIT License

## References

- NEAT-Python: [GitHub](https://github.com/CodeReclaimers/neat-python)
- Pygame Documentation: [Pygame](https://www.pygame.org/docs/)
