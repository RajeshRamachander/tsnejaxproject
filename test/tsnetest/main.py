from simple_data_processor import SimpleDataProcessor


def main():

    algorithms = [
                'ntk',
                  # 'jax_tsne',
                  # 'sklearn_tsne'
                  ]
   
    for algorithm in algorithms[::-1]:
        print(f"Starting t-SNE with {algorithm} algorithm.")
        data_processor = SimpleDataProcessor(algorithm,
                                             preparation_method='full',
                                             filename_to_save_output = None,
                                             size = 500,
                                             perplexity=30,     
                                             num_iterations=1000, 
                                             learning_rate=100,
                                             )
        print(f"Total execution time: {data_processor.start_task()} seconds")


if __name__ == '__main__':
    main()
