class EarlyStopping:

    def __init__(
        self, 
        patience: int=10, 
        verbose: int=False, 
        delta: int=0, 
    ):
        self.best_score = float("inf")
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta

    def __call__(self, score):

        if score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.early_stop = False
            self.best_score = score
            self.counter = 0