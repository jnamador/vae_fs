"""This script simply holds the base parameters to train the VAE-GAN architecture. Parameters are then overriden at the module level"""
# Base parameters ----------
NUM_TRAIN      = 10 # Number of iterations to train for.
# VAE Architecture
INPUT_SZ       = 57
H1_SZ          = 32 # Hidden layer 1 size
H2_SZ          = 16 # "          " 2 "  "
LATENT_SZ      = 3
# Discriminator Architecture # 8, 2 is on ATLAS-VAE-GAN
DISC_H1_SZ     = 8 # Size of first hidden layer of discriminator  
DISC_H2_SZ     = 2 # "" second hidden layer ""
# Training schedule and parameters
NUM_EPOCHS     = 100
STEPS_EPOCH    = 20 # Steps per epoch
BATCH_SIZE     = 1024
STOP_PATIENCE  = 40
LR_PATIENCE    = 20
LR             = 0.001 # Learning rate
REDUCE_LR_FACTOR = 0.5
VAL_SPLIT      = 0.2 # Validation split
CYCLE_LEN      = 20
SHUFFLE_BOOL   = True
# Hyperparameters
MIN_BETA       = 0
MAX_BETA       = 1
MIN_GAMMA      = 1
MAX_GAMMA      = 50
# ---

def print_base_params():
    print("""CONSTANTS IMPORTED:
            NUM_TRAIN      = 10 # Number of iterations to train for.
            # VAE Architecture
            INPUT_SZ       = 57
            H1_SZ          = 32 # Hidden layer 1 size
            H2_SZ          = 16 # "          " 2 "  "
            LATENT_SZ      = 3
            # Discriminator Architecture # 8, 2 is on ATLAS-VAE-GAN
            DISC_H1_SZ     = 8 # Size of first hidden layer of discriminator  
            DISC_H2_SZ     = 2 # "" second hidden layer ""
            # Training schedule and parameters
            NUM_EPOCHS     = 100
            STEPS_EPOCH    = 20 # Steps per epoch
            BATCH_SIZE     = 1024
            STOP_PATIENCE  = 40
            LR_PATIENCE    = 20
            LR             = 0.001 # Learning rate
            REDUCE_LR_FACTOR = 0.5
            VAL_SPLIT      = 0.2 # Validation split
            CYCLE_LEN      = 20
            SHUFFLE_BOOL   = True
            # Hyperparameters
            MIN_BETA       = 0
            MAX_BETA       = 1
            MIN_GAMMA      = 1
            MAX_GAMMA      = 50""")