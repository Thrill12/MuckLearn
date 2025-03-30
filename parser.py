from sklearn.preprocessing import StandardScaler
import numpy as np

# Assumes the outputs are on the right side of the data
# ignoreLeftColumns is used to ignore the first n columns from the left
def parseData(path, numOfInputs, numOfOutputs, ignoreLeftColumns=0):
    data = []
    
    with open(path, 'r', encoding='utf-8-sig') as file:
        for line in file:
            data.append(line.strip().split(','))

    inputs = []
    outputs = []

    # Loop over each line and convert the values to floats
    for i in range(len(data)):          
        try:
            # Convert the input values to float
            inputs.append([float(x) for x in data[i][ignoreLeftColumns:ignoreLeftColumns + numOfInputs]])
            # Convert the output values to float
            outputs.append([float(x) for x in data[i][ignoreLeftColumns + numOfInputs:ignoreLeftColumns + numOfInputs + numOfOutputs]])
        except ValueError:
            print("Error in line", i)
            return None, None

    # Flatten outputs if needed
    outputs_flat = [item for sublist in outputs for item in sublist]

    inputs, outputs_flat = remove_outliers(inputs, outputs_flat)

    return inputs, outputs_flat

# Function to remove any rows which have outliers in any input column
def remove_outliers(inputs, outputs):
    # Convert the inputs to a numpy array
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    inputs_length = len(inputs)

    # Calculate the z-scores of the inputs
    z_scores = np.abs((inputs - np.mean(inputs, axis=0)) / np.std(inputs, axis=0))

    # Remove any rows which have a z-score greater than 3
    # These values are classified as outliers
    rows_to_remove = np.any(z_scores > 3, axis=1)
    inputs = inputs[~rows_to_remove]
    outputs = outputs[~rows_to_remove]

    print("Removed", inputs_length - len(inputs), "rows with outliers")

    return inputs, outputs

# Find the correlation between each input column and the output
def get_correlations(inputs, outputs):
    # Convert the inputs to a numpy array
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    input_columns = inputs.shape[1]
    correlations = []

    # Find the correlation between each input column and the output
    for i in range(input_columns):
        correlation = np.corrcoef(inputs[:, i], outputs)[0, 1]
        correlations.append(correlation)

    return correlations