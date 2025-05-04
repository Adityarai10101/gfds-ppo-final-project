# GFDS PPO Final Project

First, create a python virtual environment. Then, run:

```bash
pip install -r requirements.txt
```

## Training

To train a new model, just run the train.py script. 

```bash
python train.py
```

After training, the results (including logs and model checkpoints) will be saved in a newly created `experiments` directory.

## Running the pretrained model

To run the model we trained, just run the following:

```bash
python run.py
```

From the root directory of this git repo