## Installation 

1. Create a Conda env that contains Python 3.8

```angular2html
conda create -n marl_venv python=3.8
```

2. Active the env
```angular2html
source activate marl_venv
```

3. Install the requirements 
```angular2html
pip install -r requirements.txt
```


## Running the Maze

Use the following command to run the maze solver interface: 
```angular2html
python main.py
```
To start the maze solver, press the SPACE bar. 



## File Structure 
```
marl/
│
├── main.py               # Entry point: sets up Pygame and runs the loop 
├── maze.py               # Maze generation and utility functions 
├── agent.py              # Agent class (pathfinding, movement, drawing)
├── solver.py             # A* algorithm implementation
└── config.py             # Settings like colors, cell size, rows, cols, etc.
```
