from simple_data_processor import SimpleDataProcessor
from mnist import  MNISTDataProcessor


def main():

    algorithms = [
                'ntk',
                  'jax_tsne',
                  'sklearn_tsne'
                  ]

    Processor = MNISTDataProcessor
   
    for algorithm in algorithms[::-1]:
        print(f"Starting t-SNE with {algorithm} algorithm.")
        data_processor = Processor(algorithm,
                                             preparation_method='full',
                                             filename_to_save_output = None,
                                             size = 1000,
                                             perplexity=30,     
                                             num_iterations=1000, 
                                             learning_rate=100,
                                             )
        print(f"Total execution time: {data_processor.start_task()} seconds")


if __name__ == '__main__':
    main()
