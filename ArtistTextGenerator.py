from TextMatrix import TextMatrix
import numpy as np
import pandas as pd
import glob 
import sys


class ArtistTextGenerator():
    def __init__(self, filepaths):
        self.filepaths = filepaths
        self.textMatrix = TextMatrix(filepaths)
        self.textMatrix.get_text()
        self.textMatrix.build_chain()
        self.chain = self.textMatrix.chain
        self.path_start = np.random.choice(range(len(self.chain))) # Deterministic 



    def get_next_word(self, word_index):
        probabilities = self.textMatrix.path_probabilities(word_index)
        next_word = np.random.choice(range(len(self.chain)), p = probabilities)
        return next_word
        

    def build_text(self, length):
        path_start = np.random.choice(range(len(self.chain))) # Randomy choose startc
        i = 0
        text = []
        curr_word = path_start
        while i < length:
            text.append(self.textMatrix.index_to_word[curr_word])
            next_word = self.get_next_word(curr_word)
            curr_word = next_word
            
            i = i + 1
        
        return " " .join(text)
    





if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    paths =  glob.glob("/Users/daniellancet/Desktop/Projects/Markov_Chain_Beginner_Project/Song-Lyrics/Kendrick-Lamar/*")
    generator = ArtistTextGenerator(paths)
    text =generator.build_text(500)
    print(text)
    # how t
    #print(generator.chain)

        





            
        





    




    
    



    




    
    




   



