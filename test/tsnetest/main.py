import time
import asyncio
import nest_asyncio
from server_communicator import ServerCommunicator
from simple_data_processor import SimpleDataProcessor

# Apply the nest_asyncio patch
nest_asyncio.apply()


async def async_output_data_processor(data_processor, processed_result):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, data_processor.output_data_processor, processed_result)


async def main_async():
    server_communicator = ServerCommunicator()
    algorithms = [
                'ntk', 
                  'jax_tsne',
                #   'sklearn_tsne'
                  ]
   
    for algorithm in algorithms[::-1]:
        print(f"Starting t-SNE with {algorithm} algorithm.")
        data_processor = SimpleDataProcessor(algorithm, 
                                             size = 1500,
                                             perplexity=30,     
                                             num_iterations=1000, 
                                             learning_rate=100
                                             )

        main_data = data_processor.prepare_data_full()
        
        response = server_communicator.start_task(main_data)

        if response.status_code == 202:
            task_id = response.json()['task_id']
            print(f'Task started with ID: {task_id}')

            start_time = time.time()
            processed_result = data_processor.wait_for_completion(task_id, server_communicator)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Total execution time: {execution_time} seconds")

            # Run the output_data_processor asynchronously
            await async_output_data_processor(data_processor, processed_result)
        else:
            print(f'Error starting the task: {response.status_code}')
        print("\n")

# Directly call the asynchronous main function without needing to get the event loop
if __name__ == '__main__':
    asyncio.run(main_async())
