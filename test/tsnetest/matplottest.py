


import matplotlib.pyplot as plt
import numpy as np

# Generate some test data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.plot(x, y, label='Sine Wave')

# Add titles and labels
plt.title('Simple Sine Wave Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Display the plot
plt.show()

print('I ran')

