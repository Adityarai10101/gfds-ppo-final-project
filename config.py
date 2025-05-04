import torch

class Config:
    MAX_STEPS = 500
    
    TIMESTEPS = 450_000
    BATCH_SIZE = 2048
    LEARNING_RATE = 3e-4
    CLIP_RATIO = 0.2
    ENTROPY_COEF = 0.01
    EPOCHS_PER_UPDATE = 10
    
    LOG_FREQ = 10      
    VIDEO_FREQ = 200    
    CHECKPOINT_FREQ = 200

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.torch.backends.mps.is_available() 
        else "cpu"
    )