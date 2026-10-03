# AI-Pong v1.3

A Neural Network based Genetic Algorithm finds the optimal solution for playing 2d game, pong.

## Change log:
+ Migrated from `pygame` library to `Arcade` library.
+ `numpy` based perceptron model has been replaced with `torch.NN` module based model.
+ Neural Net model structure has been changed.
+ Updated fitness function.
+ Added visual representation of the Neural Network to the UI.
+ Visual improvements (ball and paddle have borders)

### Performance improvements
+ Entire population's neural networks run in batch, resulting in performance boost.
+ Only elite player's playzone is displayed on the screen.
+ A pre-defined number of frames can be skipped before the next frame is rendered.


+ save each GA run (generational models, logs, possibly settings into a unique place,
possibly as a folder named by the datetime),
e.g. If we run with population size 100 on 01.01.1999 4:53AM, we save in folder 01-01-99-04-53-AM where,
population size is written in a log, along with any other data

## Installation:
`pip install -r requirements.txt`

## FAQ:
**Q: Scores don't go above a certain threshold.**

**A: Update `'TIME_OUT': 12` in `src/variables.py` with a value of your choice. 
This will make a generation last longer.**

### Note:
The changelog may not accurately represent changes and, some changes may not be listed.

