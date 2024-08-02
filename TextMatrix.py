import pandas as pd
import numpy as np
import glob
import string


class TextMatrix():
    def __init__(self, text_paths):
        self.text_paths = text_paths
        self.chain = None
        self.text = None
        self.index_to_word = None

    def get_text(self):
        corpus = []
        for path in self.text_paths:
            with open(path, "r") as my_file:
                data = my_file.read()
                data = data.translate(str.maketrans('', '', string.punctuation))
                data = data.replace("/", " ").replace("\\", " ").replace("\n", " ")
                data_into_series = pd.Series(data.lower().split())
                corpus.append(data_into_series)
        
        # Concatenate corpus
        self.text = np.array(pd.concat(corpus, axis=0))
        # Remove empty strings
        self.text = self.text[self.text != '']

    def build_chain(self):
        unique_words = set(self.text)
        dimension = len(unique_words)
        index_dict = {value: idx for idx, value in enumerate(unique_words)}
        self.chain = np.zeros((dimension, dimension), dtype=int)
        self.index_to_word = dict((v, k) for k, v in index_dict.items())

        for l in range(len(self.text) - 1):
            word_from = index_dict[self.text[l]]
            word_to = index_dict[self.text[l + 1]]
            self.chain[word_from, word_to] += 1  # Corrected order

    def path_probabilities(self, word_index):
        row_sum = self.chain[word_index].sum()
        if row_sum == 0:
            weights = np.full(len(self.chain), 1 / len(self.chain))
        else:
            weights = self.chain[word_index] / row_sum
        return weights




if __name__ == "__main__":
  



#  import matplotlib.pyplot as plt
#  import networkx as nx
#  import glob
    paths =  glob.glob("/Users/daniellancet/Desktop/Projects/Markov_Chain_Beginner_Project/Song-Lyrics/Kendrick-Lamar/3117.txt")
    matrix_obj = TextMatrix(paths)

    matrix_obj.get_text()
    matrix_obj.build_chain()

    print(matrix_obj.index_to_word)
   
 #matrix

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