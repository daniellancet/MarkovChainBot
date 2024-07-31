from TextMatrix import TextMatrix
import numpy as np
class ArtistTextGenerator():
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self.textMatrix = TextMatrix(filepaths)
        self.textMatrix.get_text()
        self.textMatrix.get_chain()
        self.chain = self.textMatrix.chain
        self.path_start = np.random.choice(range(len(self.chain)), random_state = 10) # Deterministic 
    




    
    




   



