from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import ast
import pandas as pd
from base_data_processor import BaseDataProcessor

class SimpleDataProcessor(BaseDataProcessor):

    def load_data(self):

        try:
            digits, digit_class = load_digits(return_X_y=True)
            if self.size is None or self.size > digits.shape[0]:
                # Use the full dataset if size is None or larger than the dataset
                self.data = digits
                self.classes = digit_class
                self.size = digits.shape[0]
            else:
                # Otherwise, randomly select a subset of the data
                rand_idx = np.random.choice(np.arange(digits.shape[0]), size=self.size, replace=False)
                self.data = digits[rand_idx, :]
                self.classes = digit_class[rand_idx]
                self.size = digits.shape[0]
            print(f"Data prepared with shape: {self.data.shape}")
        except Exception as e:
            print(f"Error preparing data: {e}")
            raise ValueError(f"load data failed")

        return self.data, self.classes, self.size


    def prepare_data(self):
        try:
            if self.preparation_method == 'full':
                transmit_data = {
                    'packer' : 'full',
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
            # print(f'Result: {processed_result}')
            print("Processing result prepared with the full method.")

        if isinstance(processed_result, list):
            low_dim = np.array(processed_result)
        else:
            # If it's a string, perform the original processing
            result_received = processed_result.replace("Result received: ", "")
            low_dim = ast.literal_eval(result_received)
            low_dim = np.array(low_dim)

        rcParams["font.size"] = 18
        rcParams["figure.figsize"] = (12, 8)

        print(f"Using {self.algorithm} for t-SNE.")
        scatter = plt.scatter(low_dim[:, 0], low_dim[:, 1], cmap="tab10", c=self.classes)
        plt.legend(*scatter.legend_elements(), fancybox=True, bbox_to_anchor=(1.05, 1))
        # Add text indicating the type of matrix or algorithm used
        plt.text(x=0.5, y=-0.1, s=f"Algo Type: {self.algorithm}", fontsize=12, ha='center', va='bottom',
                 transform=plt.gca().transAxes)

        plt.show()


    def output_data_processor_matrix(self, processed_result):
    

        
        if processed_result is None:
            print("No result to process.")
            return
        else:
            # print(f'Result: {processed_result}')
            print("Processing result prepared with the matrix method.")

    
        if isinstance(processed_result, list):
            matrix = np.array(processed_result)
        else:
            # If it's a string, perform the original processing
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

        # Add text indicating the type of matrix or algorithm used
        plt.text(x=0.5, y=-0.1, s=f"Matrix Type: {self.algorithm}", fontsize=12, ha='center', va='bottom', transform=plt.gca().transAxes)

        # Customizing the ticks (optional, depending on your dataset)
        plt.xticks(np.arange(matrix.shape[1]))  # Adjust as necessary
        plt.yticks(np.arange(matrix.shape[0]))  # Adjust as necessary

        plt.show()

        # save data to csv file
        pd.DataFrame(matrix).to_csv(self.filename_to_save_output, index=False)

