from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import ast
from base_data_processor import BaseDataProcessor

class MNISTDataProcessor(BaseDataProcessor):

    def load_data(self):
        try:
            X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
            X = X.to_numpy()
            y = y.astype(int).to_numpy()  # Convert labels to integers and to numpy array

            if self.size is None or self.size > X.shape[0]:
                # Use the full dataset if size is None or larger than the dataset
                self.data = X
                self.classes = y
                self.size = X.shape[0]
            else:
                # Otherwise, randomly select a subset of the data
                rand_idx = np.random.choice(np.arange(X.shape[0]), size=self.size, replace=False)
                self.data = X[rand_idx, :]
                self.classes = y[rand_idx]
                self.size = self.size

            print(f"Data prepared with shape: {self.data.shape}")
        except Exception as e:
            print(f"Error preparing data: {e}")
            raise ValueError("load data failed")

        return self.data, self.classes, self.size

    def prepare_data(self):
        try:
            if self.preparation_method == 'full':
                transmit_data = {
                    'packer': 'full',
                    'data': self.data.tolist(),
                    'num_dimensions': 2,
                    'perplexity': self.perplexity,
                    'num_iterations': self.num_iterations,
                    'learning_rate': self.learning_rate,
                    'algorithm': self.algorithm,
                }
                return transmit_data
            else:
                transmit_data = {
                    'packer': 'matrix',
                    'data': self.data.tolist(),
                    'algorithm': self.algorithm,
                }
                return transmit_data
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None

    def output_data_processor(self, processed_result):
        if self.preparation_method == 'full':
            self.output_data_processor_full(processed_result)
        elif self.preparation_method == 'matrix':
            self.output_data_processor_matrix(processed_result)
        else:
            print(f"No output processor defined for preparation method '{self.preparation_method}'.")

    def output_data_processor_full(self, processed_result):
        if processed_result is None:
            print("No result to process.")
            return
        else:
            print("Processing result prepared with the full method.")

        if isinstance(processed_result, list):
            low_dims_frames = np.array(processed_result)
            print(f'low dimensions: {low_dims_frames}')
        else:
            result_received = processed_result.replace("Result received: ", "")
            low_dims_frames = ast.literal_eval(result_received)
            low_dims_frames = np.array(low_dims_frames)
            print(f'frames: {low_dims_frames}')

        rcParams["font.size"] = 18
        rcParams["figure.figsize"] = (12, 8)

        if self.algorithm != 'sklearn_tsne':
            low_dim = low_dims_frames[-1]
        else:
            low_dim = low_dims_frames

        if low_dim.shape[0] != len(self.classes):
            print("Warning: The number of data points and the number of classes do not match. Adjusting the class labels.")
            classes = self.classes[:low_dim.shape[0]]
        else:
            classes = self.classes

        print(f"Using {self.algorithm} for t-SNE.")
        scatter = plt.scatter(low_dim[:, 0], low_dim[:, 1], cmap="tab10", c=classes)
        plt.legend(*scatter.legend_elements(), fancybox=True, bbox_to_anchor=(1.05, 1))

        plt.text(x=0.5, y=-0.1, s=f"Algo Type: {self.algorithm}", fontsize=12, ha='center', va='bottom',
                 transform=plt.gca().transAxes)

        plt.show()

    def output_data_processor_matrix(self, processed_result):
        if processed_result is None:
            print("No result to process.")
            return
        else:
            print("Processing result prepared with the matrix method.")

        if isinstance(processed_result, list):
            matrix = np.array(processed_result)
        else:
            result_received = processed_result.replace("Result received: ", "")
            try:
                matrix = ast.literal_eval(result_received)
                matrix = np.array(matrix)
            except ValueError as e:
                print(f"Error parsing string to array: {e}")
                return

        cmap = 'viridis'
        figsize = (10, 8)
        plt.figure(figsize=figsize)
        plt.imshow(matrix, cmap=cmap, aspect='auto')
        plt.colorbar()  # Show color scale

        plt.title('Distance Matrix Heatmap')
        plt.xlabel('Data Point Index')
        plt.ylabel('Data Point Index')

        plt.text(x=0.5, y=-0.1, s=f"Matrix Type: {self.algorithm}", fontsize=12, ha='center', va='bottom', transform=plt.gca().transAxes)

        plt.xticks(np.arange(matrix.shape[1]))  # Adjust as necessary
        plt.yticks(np.arange(matrix.shape[0]))  # Adjust as necessary

        plt.show()

        pd.DataFrame(matrix).to_csv(self.filename_to_save_output, index=False)



# # Test the MNISTDataProcessor class
#
# data_processor = MNISTDataProcessor(algorithm='ntk',
#                                              preparation_method='full',
#                                              filename_to_save_output = None,
#                                              size = 1000,
#                                              perplexity=30,
#                                              num_iterations=1000,
#                                              learning_rate=100,
#                                              )
# data, classes, size = data_processor.load_data()
#
# print("Data shape:", data.shape)
# print("Classes shape:", classes.shape)
# print("Size:", size)
#
# print(f"Total execution time: {data_processor.start_task()} seconds")
