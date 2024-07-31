import pandas as pd
import numpy as np
import glob


class TextMatrix():
    def __init__(self, text_paths):
        self.text_paths = text_paths
        self.chain = None
        self.text = None
    
    
    def get_text(self):
        corpus = []
        for path in self.text_paths:
            my_file = open(path, "r") 

            data = my_file.read() 
            data_into_series = pd.Series((data.replace('\n', ' ').replace(",", "").replace('"', " ").replace(":", " ").
                                          replace("&", "and").replace("(", " ").replace(")", " ").
                                          replace("?", " ?").replace("!", " !").replace("/", " ").replace("]", "").replace("[", "").lower().split(' ')))
            

            corpus.append(data_into_series)
        
        # concat corpus
        self.text =  np.array(pd.concat(corpus, axis = 0))

        
        


    
    def build_chain(self):
        unique_words = set(self.text)
        
        dimension = len(unique_words)
        index_dict = {value: idx for idx, value in enumerate(unique_words)}
        self.chain = np.zeros((dimension, dimension), dtype = int)

        l, r = 0, 1

        while r < len(unique_words):
            word_from = index_dict[self.text[l]]
            word_to = index_dict[self.text[r]]
            self.chain[word_to, word_from] =  self.chain[word_to, word_from] + 1
           
            r = r + 1
            l = l + 1
        





if __name__ == "__main__":
   print("hello")



#  import matplotlib.pyplot as plt
#  import networkx as nx
#  import glob
#  paths =  glob.glob("/Users/daniellancet/Desktop/Projects/Markov_Chain_Beginner_Project/Song-Lyrics/Kendrick-Lamar/*")
#  matrix_obj = TextMatrix(paths)

#  matrix_obj.get_text()
#  matrix_obj.build_chain()

#  matrix = matrix_obj.chain
#  index_dict = {value: idx for idx, value in enumerate(set(matrix_obj.text))}


# # # Create a NetworkX graph
# G = nx.DiGraph()

# # Add nodes with labels
# for value, idx in index_dict.items():
#     G.add_node(idx, label=value)

# # Add edges with weights from the matrix
# for i in range(matrix.shape[0]):
#     for j in range(matrix.shape[1]):
#         if matrix[i, j] > 10:
#             G.add_edge(j, i, weight=matrix[i, j])

# nodes_to_remove = [node for node in G.nodes if G.degree(node) < 2]
# G.remove_nodes_from(nodes_to_remove)


# # Create a layout for the graph
# pos = nx.spring_layout(G, seed=42)  # Seed for reproducibility

# # Draw the nodes with labels
# labels = nx.get_node_attributes(G, 'label')
# nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue', alpha=0.8)
# nx.draw_networkx_labels(G, pos, labels, font_size=3)

# # Draw the edges with weights
# edges = G.edges(data=True)
# nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='-|>', arrowsize=3, width=1)
# edge_labels = {(u, v): f'{d["weight"]}' for u, v, d in edges}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=5)

# # Display the plot
# plt.title('Co-occurrence Graph')
# plt.show()