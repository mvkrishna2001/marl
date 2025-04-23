## Installation 

1. Create a Conda env that contains Python 3.10

```angular2html
conda create -n marl_venv python=3.10
```

2. Active the env
```angular2html
source activate marl_venv
```

3. Install the requirements 
```angular2html
pip install -r requirements.txt
```

4. Install ffmpeg for maze visualizations 
```angular2html
sudo apt install ffmpeg
```

## Running the Maze

Use the following command to run the maze solver interface: 
```angular2html
python main.py
```
To start the maze solver, press the SPACE bar.

### Arguments 
To run the solver with a specific algorithm, append `--algorithm <algorithm>`. Choices include `['a_star', 'bfs', 'dfs', 'dijkstra']`. 

Example of use:
```angular2html
python main.py --algorithm a_star
```

## File Structure 
```
marl/
│
├── main.py               # Entry point: sets up Pygame and runs the loop 
├── maze.py               # Maze generation and utility functions 
├── agent.py              # Agent class (pathfinding, movement, drawing)
├── solver.py             # Algorithm implementations
└── config.py             # Settings like colors, cell size, rows, cols, etc.
```


## Notes 
The gym maze environment code is taken from https://github.com/rpinsler/gym-maze/tree/master, with minor modifications. 
