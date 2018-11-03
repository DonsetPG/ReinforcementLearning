

class Parameters:

    def __init__(self):
        self.T_MAX = 7000000
        self.NUM_THREADS = 16
        self.INITIAL_LEARNING_RATE = 7e-4
        self.DISCOUNT_FACTOR = 0.99
        self.VERBOSE_EVERY = 40000
        self.TESTING = False
        self.I_ASYNC_UPDATE = 5
        self.LR = 7e-4
        self.PROBA_HER = 4
        self.LAMBDA = 0.95
        self.C_ENTROPY = 0.01
        self.C_LOSS = 0.5
        self.MAX_GRAD = 1.0
        self.N_EPOCHS = 4
        self.N_BATCH = 32
        self.CLIP = 0.2
        self.FREQ_SAVES = 100
        self.N_FRAMES = 1e6
        self.USE_GAE = True
        self.ALPHA = 1



def get_args():

    arg = Parameters()
    return arg


def get_args_ppo():

    arg = Parameters()
    arg.I_ASYNC_UPDATE = 128
    arg.C_LOSS = 1
    arg.N_EPOCHS = 3
    arg.N_BATCH = 32*8
    arg.NUM_THREADS = 16
    arg.CLIP = 0.2
    return arg
