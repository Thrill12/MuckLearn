import parser
import numpy as np
import os
from NeuralNetwork import NeuralNetwork
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import matplotlib.pyplot as plt

def main():

    num_inputs = 8
    num_to_ignore = 1
    learning_rate = 0.15
    hidden_layers = [300,300]
    momentum = 0.3
    epochs = 1000
    log_loops = 5
    batch_size = 64
    base_name = "Model A"

    # Parsing inputs and outputs
    inputs, outputs = parser.parseData("model a.csv", num_inputs, 1, num_to_ignore)

    if inputs is None or outputs is None:
        print("Error parsing data")
        return

    try_for_best_hidden_layer = False
    # Min/Max Values for the trial for the best hidden layer
    min_trial, max_trial = 4, 20

    # Create a dictionary to store the name and accuracy
    name_accuracy_dict = {}
    name_accuracy_list = []
    
    # Create a boolean to check if the average should be calculated for the graph
    # If True, the graph will show the average of the accuracy for each number of hidden layer in the trial
    # If False, the graph will show the individual accuracy for each iteration
    average = False

    log_path = "logs"

    if try_for_best_hidden_layer:
        # loop to try the best hidden layer
        for layer in range (min_trial, max_trial + 1):
            hidden_layers = [layer]
            name = base_name + " " + str(hidden_layers)
            # Reset matplotlib
            plt.clf()

            average = True

            log_path = train_loop(log_loops, learning_rate, hidden_layers, momentum, epochs, inputs, outputs, name, name_accuracy_list, num_inputs, base_name, batch_size)
    else:
        average = False

        log_path = train_loop(log_loops, learning_rate, hidden_layers, momentum, epochs, inputs, outputs, base_name, name_accuracy_list, num_inputs, base_name, batch_size)        

    # Plot the values for the name_accuracy_dict in a bar chart
    plt.clf()

    # Work out average for the dictionary
    for name, accuracy, accuracy_15, accuracy_20 in name_accuracy_list:
        if name in name_accuracy_dict:
            name_accuracy_dict[name] += accuracy
        else:
            name_accuracy_dict[name] = accuracy

    # Sort the dictionary by values
    name_accuracy_dict = dict(sorted(name_accuracy_dict.items()))

    # if average, plot the dictionary
    if average:
        plt.clf()

        for name in name_accuracy_dict:
            name_accuracy_dict[name] /= log_loops

        plt.bar(name_accuracy_dict.keys(), name_accuracy_dict.values())
        plt.xticks(rotation=90)
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy of Neural Networks with Different Hidden Layers")
        plt.tight_layout()

        if not os.path.exists(base_name):
            os.makedirs(base_name)

        plt.savefig(base_name + "/" + "best_structure_trial.png")    
    
def log_nn(log_lines, current_log, nn_name, nn, epochs, training_time, cumulative_errors, accuracy):
    log_name = "log_" + str(current_log)
    log_folder = nn_name
    log_path = log_folder + "/" + log_name

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    with open(log_path + ".txt", "w") as file:
        file.write("Hyperparameters\n")
        file.write("Final Learning rate: " + str(nn.learning_rate) + "\n")
        file.write("Layer Structure: " + str(nn.layers) + "\n")
        file.write("Momentum: " + str(nn.momentum) + "\n")
        file.write("Iterations: " + str(epochs) + "\n")
        file.write("Training time: " + str(training_time) + "\n")
        file.write("Final RMSE: " + str(nn.last_rmse) + "\n")
        file.write("Accuracy: " + str(accuracy) + "\n")
        file.write("\n")
        file.write("Runtime log: " + "\n")
        for line in log_lines:
            file.write(str(line) + "\n")

    plt.clf()
    print("Graphs")
    plt.plot(range(len(cumulative_errors)), np.array(cumulative_errors))
    plt.xlabel("Iteration")
    plt.ylabel("Error (RMSE of standardised data)")
    plt.savefig(log_path + "_error.png")

def train_loop(log_loops, learning_rate, hidden_layers, momentum, epochs, inputs, outputs, name, name_accuracy_list, num_of_inputs, base_name, batch_size):
    loop_specific_accuracy_list = []
    rmse_list = []
    acc_10_list = []
    acc_15_list = []
    acc_20_list = []
    training_time_list = []
    correlation_list = []

    log_folder = base_name + "/" + name

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    cumulative_errors_graphs = []
    for i in range(log_loops):
        nn = NeuralNetwork(learning_rate, num_of_inputs, hidden_layers, momentum, name)

        # Create separate scalers for inputs and outputs
        input_scaler = StandardScaler()
        output_scaler = StandardScaler()

        # Need to split data into train/test/val sets
        all_input_vectors = np.array(inputs, dtype=np.float64)
        all_targets = np.array(outputs)

        # Shuffle all input vectors and all targets
        indices = np.random.permutation(len(all_input_vectors))
        all_input_vectors = all_input_vectors[indices]
        all_targets = all_targets[indices]

        # Standardize inputs
        all_input_vectors = input_scaler.fit_transform(all_input_vectors)

        # Standardize targets (reshape targets to a 2D array first)
        all_targets = output_scaler.fit_transform(all_targets.reshape(-1, 1)).flatten()

        # Split train/test/val into 
        train_size = int(len(all_input_vectors) * 0.75)
        test_size = int(len(all_input_vectors) * 0.25)

        train_input_vectors = all_input_vectors[:train_size]
        train_targets = all_targets[:train_size]

        test_input_vectors = all_input_vectors[train_size:train_size + test_size]
        test_targets = all_targets[train_size:train_size + test_size]

        val_input_vectors = all_input_vectors[train_size + test_size:]
        val_targets = all_targets[train_size + test_size:]
        
        print("Training...")

        epoch_predictions = []

        # Train on the training data
        cumulative_errors, log, training_time, val_rmses = nn.train(train_input_vectors, train_targets, epochs, input_scaler, output_scaler, batch_size, val_input_vectors, val_targets)
        cumulative_errors_graphs.append(cumulative_errors)  

        print("VALIDATIONS: ", val_rmses)

        # Plot Validation RMSE
        plt.clf()
        plt.plot(range(len(val_rmses)), np.array(val_rmses))
        plt.xlabel("Iteration")
        plt.ylabel("Error (RMSE)")
        plt.title("Validation RMSE")
        plt.savefig(log_folder + "/log_" + str(i) + "_validation.png")

        # have a list of transformed test targets
        transformed_test_targets = output_scaler.inverse_transform(test_targets.reshape(-1, 1)).flatten()

        total_test = len(test_input_vectors)
        correct = 0
        correct_15 = 0
        correct_20 = 0

        predictions = []
        targets = []

        log.append("Seed used: " + str(nn.seed))
        log.append("\nTesting with the validation data (" + str(len(test_input_vectors)) + " samples)")
        
        relative_errors = []

        final_predictions = []
        for j in range(len(test_input_vectors)):
            # Test on the test data using the _predict method
            prediction = nn._predict(test_input_vectors[j], output_scaler)[0]
            predictions.append(prediction)
            targets.append(transformed_test_targets[j])
            # Print the predicted and actual values
            difference =  prediction - transformed_test_targets[j]
            actual_value = transformed_test_targets[j]

            correct_bool = False
            # Check if the prediction is within 10% of the actual value
            if np.abs(difference) < (actual_value * 0.1):
                correct += 1
                correct_bool = True

            if np.abs(difference) < (actual_value * 0.15):
                correct_15 += 1

            if np.abs(difference) < (actual_value * 0.2):
                correct_20 += 1

            pred_str = str(correct_bool) + " Predicted: " + str(np.round(prediction, 1)) + " Actual: " + str(np.round(actual_value, 1)) + " Difference: " + str(np.round(difference, 1))+ " (" + str(np.round((difference / actual_value) * 100, 1)) + "%)"
            final_predictions.append(pred_str)

            

        rmse = np.sqrt(np.mean((np.array(predictions) - np.array(targets)) ** 2))
        log.append("Real RMSE: " + str(rmse))

        # Calculate the correlation coefficient between the predictions and the targets
        correlation = np.corrcoef(targets, predictions)[0, 1]
        log.append("Correlation: " + str(correlation))
        print("Correlation: ", correlation)

        accuracy = np.round((correct / total_test) * 100, 2)
        print("Accuracy: ", accuracy, "% (within 10%)")
        log.append("Accuracy: " + str(accuracy) + "% (within 10%)")

        accuracy_15 = np.round((correct_15 / total_test) * 100, 2)
        print("Accuracy: ", accuracy_15, "% (within 15%)")
        log.append("Accuracy: " + str(accuracy_15) + "% (within 15%)")

        accuracy_20 = np.round((correct_20 / total_test) * 100, 2)
        print("Accuracy: ", accuracy_20, "% (within 20%)")
        log.append("Accuracy: " + str(accuracy_20) + "% (within 20%)")

        # Set the name_accuracy_dict
        name_accuracy_list.append((name, accuracy, accuracy_15, accuracy_20))
        loop_specific_accuracy_list.append((name,accuracy, accuracy_15, accuracy_20))

        rmse_list.append(nn.last_rmse)
        acc_10_list.append(accuracy)
        acc_15_list.append(accuracy_15)
        acc_20_list.append(accuracy_20)
        training_time_list.append(training_time)
        correlation_list.append(correlation)

        # Save a scatter plot of the predictions against the targets
        

        correlations = parser.get_correlations(all_input_vectors, all_targets)

        

        # Plot bar graph of correlations
        plt.clf()
        plt.bar(range(len(correlations)), correlations)
        plt.xticks(range(len(correlations)), range(len(correlations)))
        plt.xlabel("Input Column")
        plt.ylabel("Correlation")
        plt.title("Correlation between input columns and output")
        plt.savefig(log_folder + "/correlation")
        if log_loops == 1:
            plt.show()

        # Find columns with a correlation coefficient of less than 0.3
        low_correlation_columns = [i for i, x in enumerate(correlations) if x < 0.3]
        print("Columns with low correlation: ", low_correlation_columns)
        log.append("Columns with low correlation: " + str(low_correlation_columns) + "\n")

        # Append all the final predictions to the log
        log.append("Final Predictions:")
        for pred in final_predictions:
            log.append(pred)

        # Find max from either the targets or the predictions
        max_val = max(np.max(targets), np.max(predictions)).flatten()[0]
        line = [0, max_val]

        # Sort predictions and targets in order of targets from small to high
        sorted_predictions = [x for _, x in sorted(zip(predictions, targets))]
        sorted_targets = sorted(predictions)

        # Figure out lines of lower and upper boundaries of the target (10%)
        lower_bound = [x - (x * 0.1) for x in sorted_targets]
        upper_bound = [x + (x * 0.1) for x in sorted_targets]

        plt.clf()
        plt.scatter(targets, predictions)
        plt.plot(line, line, color='red', linestyle='dashed')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        title = "Predicted vs Actual [" + str(np.round(correlation, 2)) + "]"
        plt.title(title)
        plt.savefig(log_folder + "/log_" + str(i) + "_scatter")

        # Plot graph of log of prediction vs actual
        plt.clf()
        plt.plot(sorted_predictions, label="Predicted")
        plt.plot(sorted_targets, label="Actual")

        # Plot the bounds on the graph
        plt.plot(lower_bound, label="Lower Bound", linestyle='dashed')
        plt.plot(upper_bound, label="Upper Bound", linestyle='dashed')

        plt.legend()
        plt.xlabel("Sample")
        plt.ylabel("Value")
        plt.title("Predicted vs Actual")
        plt.savefig(log_folder + "/log_" + str(i) + "_line")

        if log_loops == 1:
            plt.show()
        # Log the results
        log_nn(log, i, log_folder, nn, epochs, training_time, cumulative_errors, accuracy)

    # Plot the accuracies for each loop
    acc_10 = [accuracy for name, accuracy, accuracy_15, accuracy_20 in loop_specific_accuracy_list]
    acc_15 = [accuracy_15 for name, accuracy, accuracy_15, accuracy_20 in loop_specific_accuracy_list]
    acc_20 = [accuracy_20 for name, accuracy, accuracy_15, accuracy_20 in loop_specific_accuracy_list]
    plt.clf()
    
    # Find the average of all of the lists then log them inside the log file
    average_rmse = np.mean(rmse_list)
    average_acc_10 = np.mean(acc_10_list)
    average_acc_15 = np.mean(acc_15_list)
    average_acc_20 = np.mean(acc_20_list)
    average_training_time = np.mean(training_time_list)
    average_correlation = np.mean(correlation_list)

    # Save the summary details inside a summary.txt file
    with open(log_folder + "/summary.txt", "w") as file:
        file.write("Average RMSE: " + str(average_rmse) + "\n")
        file.write("Average Accuracy within 10%: " + str(average_acc_10) + "\n")
        file.write("Average Accuracy within 15%: " + str(average_acc_15) + "\n")
        file.write("Average Accuracy within 20%: " + str(average_acc_20) + "\n")
        file.write("Average Training Time: " + str(average_training_time) + "\n")
        file.write("Average Correlation: " + str(average_correlation) + "\n")

    # Plot each accuracy as a bar next to each other
    x = np.arange(len(loop_specific_accuracy_list))
    width = 0.3
    plt.bar(x, acc_10, width, label="Within 10%")
    plt.bar(x + width, acc_15, width, label="Within 15%")
    plt.bar(x + width * 2, acc_20, width, label="Within 20%")

    plt.legend()
    plt.xticks(rotation=90)
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.title(base_name)
    plt.tight_layout()
    plt.savefig(log_folder + "/structure_accuracies.png")

    # Plot all of the cumulative errors graphs into one graph
    plt.clf()
    for i, errors in enumerate(cumulative_errors_graphs):
        plt.plot(range(len(errors)), np.array(errors), label="Log " + str(i))

    plt.xlabel("Iteration")
    plt.ylabel("Error (RMSE of standardised data)")
    plt.legend()
    plt.savefig(log_folder + "/cumulative_errors.png")

    return base_name

main()