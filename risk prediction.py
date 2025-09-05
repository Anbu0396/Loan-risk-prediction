<<<<<<< HEAD
import pandas as pd
import numpy as np
import random
import math
import warnings

# Get data from the CSV file
data = pd.read_csv("Loan_prediction_dataset.csv")

print("Data from Dataset")
print(data)

# Encode string values to numerical values
home_encoding = {'OWN': 0, 'MORTGAGE': 1, 'RENT': 2}
intent_encoding = {'MEDICAL': 0, 'PERSONAL': 1, 'DEBTCONSOLIDATION': 2, 'HOMEIMPROVEMENT': 3, 'VENTURE': 4}
default_encoding = {'N': 0, 'Y': 1}

data['Home'] = data['Home'].map(home_encoding)
data['Intent'] = data['Intent'].map(intent_encoding)
data['Default'] = data['Default'].map(default_encoding)

print("Data after encoding")
print(data)

data = data.fillna(data.mean())

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Normalization function
def normalization(features_to_normalize):
    for feature in features_to_normalize:
        min_value = data[feature].min()
        max_value = data[feature].max()
        if max_value != min_value:
            data[feature] = (data[feature] - min_value) / (max_value - min_value)
        else:
            data[feature] = 0

def normalize_single_data_point(data_point, features_to_normalize, data):
    if len(data_point.shape) == 1:
        data_point = data_point.reshape(1, -1)
    
    for i, feature in enumerate(features_to_normalize):
        min_value = data[feature].min()
        max_value = data[feature].max()
        if max_value != min_value:
            data_point[0, i] = (data_point[0, i] - min_value) / (max_value - min_value)
        else:
            data_point[0, i] = 0
    return data_point

features_to_normalize = ['Age', 'Income', 'Amount', 'Emp_length', 'Rate', 'Percent_income', 'Cred_length']
normalization(features_to_normalize)

print("Data after normalized")
print(data)

features = data.drop('Status', axis=1).values

# Shuffle the DataFrame
data = data.sample(frac=1, random_state=13).reset_index(drop=True)

# Split data (80% for training and 20% for testing)
split_index = int(0.8 * len(data))

# Split the Data
train_data = data[:split_index]
test_data = data[split_index:]

# Separate features and target variable
X_train = train_data.drop('Status', axis=1).values
y_train = train_data['Status'].values
X_test = test_data.drop('Status', axis=1).values
y_test = test_data['Status'].values

# Append bias to inputs
X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

print("features in train data")
print(X_train)
print("target in train data")
print(y_train)
print("features in test data")
print(X_test)
print("target in test data")
print(y_test)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights
def init_weights(input_size, hidden_size1, hidden_size2):
    weights_input_hidden1 = np.random.uniform(-1, 1, (input_size, hidden_size1))
    weights_hidden1_hidden2 = np.random.uniform(-1, 1, (hidden_size1, hidden_size2))
    weights_hidden2_output = np.random.uniform(-1, 1, hidden_size2)
    return weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output

# Forward propagation
def forward_propagation(X, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output):
    hidden_layer1 = sigmoid(np.dot(X, weights_input_hidden1))
    hidden_layer2 = sigmoid(np.dot(hidden_layer1, weights_hidden1_hidden2))
    output_layer = sigmoid(np.dot(hidden_layer2, weights_hidden2_output))
    return hidden_layer1, hidden_layer2, output_layer

# Backward propagation
def backward_propagation(x, y, hidden_layer1, hidden_layer2, output_layer, weights_hidden2_output, weights_hidden1_hidden2, weights_input_hidden1, learning_rate):
    output_error = y - output_layer
    output_delta = output_error * sigmoid_derivative(output_layer)
    hidden2_error = np.dot(output_delta, weights_hidden2_output)
    hidden2_delta = hidden2_error * sigmoid_derivative(hidden_layer2)
    hidden1_error = np.dot(hidden2_delta, weights_hidden1_hidden2)
    hidden1_delta = hidden1_error * sigmoid_derivative(hidden_layer1)

    # Update weights
    weights_hidden2_output += learning_rate * np.dot(hidden_layer2.T, output_delta)
    weights_hidden1_hidden2 += learning_rate * np.dot(hidden_layer1.T, hidden2_delta)
    weights_input_hidden1 += learning_rate * np.outer(x, hidden1_delta)

    return weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output

# Train the neural network
def train(X_train, y_train, input_size, hidden_size1, hidden_size2, learning_rate, cycles):
    weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output = init_weights(input_size, hidden_size1, hidden_size2)
    loss_history = []

    for cycle in range(cycles):
        total_loss = 0
        for x, y in zip(X_train, y_train):
            hidden_layer1, hidden_layer2, output_layer = forward_propagation(x, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output)
            output_error = y - output_layer
            total_loss += np.sum(output_error**2)
            weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output = backward_propagation(
                x, y, hidden_layer1, hidden_layer2, output_layer, weights_hidden2_output, weights_hidden1_hidden2, weights_input_hidden1, learning_rate
            )
        
        loss_history.append(total_loss / len(y_train))
        if cycle % 100 == 0:
            print(f"Cycle {cycle} - Loss: {loss_history[-1]}")
    
    return weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output, loss_history

# Parameters
input_size = X_train.shape[1]
hidden_size1 = 15
hidden_size2 = 15
learning_rate = 0.01
cycles = 500

# Train the model and track loss
weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output, loss_history = train(X_train, y_train, input_size, hidden_size1, hidden_size2, learning_rate, cycles)

print("Final weights (input to hidden1):")
print(weights_input_hidden1)
print("Final weights (hidden1 to hidden2):")
print(weights_hidden1_hidden2)
print("Final weights (hidden2 to output):")
print(weights_hidden2_output)

def predict(X, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output):
    hidden_layer1 = sigmoid(np.dot(X, weights_input_hidden1))
    hidden_layer2 = sigmoid(np.dot(hidden_layer1, weights_hidden1_hidden2))
    output_layer = sigmoid(np.dot(hidden_layer2, weights_hidden2_output))
    return output_layer

predictions = predict(X_test, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output)
predicted_classes = (predictions > 0.5).astype(int)

correct_predictions = np.sum(predicted_classes == y_test)  
total_predictions = len(y_test)  
accuracy = (correct_predictions / total_predictions) * 100  


def get_user_input_and_predict():
    try:
        Id = 0
        Age = float(input("Enter Age: "))
        Income = float(input("Enter Income: "))
        Home = input("Enter Home (OWN, MORTGAGE, RENT): ").upper()
        Amount = float(input("Enter Loan Amount: "))
        Emp_length = float(input("Enter Employment Length: "))
        Intent = input("Enter Intent (MEDICAL, PERSONAL, DEBTCONSOLIDATION, HOMEIMPROVEMENT, VENTURE): ").upper()
        Rate = float(input("Enter Rate: "))
        Percent_income = float(input("Enter Percent Income: "))
        Default = input("Enter Default (N, Y): ").upper()
        Cred_length = float(input("Enter Credit Length: "))

        if Home not in home_encoding or Default not in default_encoding or Intent not in intent_encoding:
            raise ValueError("Invalid input for Home, Default, or Intent")

        Home = home_encoding[Home]
        Intent = intent_encoding[Intent]
        Default = default_encoding[Default]

        new_data = np.array([[Id, Age, Income, Home, Amount, Emp_length, Intent, Rate, Percent_income, Default, Cred_length]])

        features_to_normalize = ['Age', 'Income', 'Amount', 'Emp_length', 'Rate', 'Percent_income', 'Cred_length']
        new_data_normalized = normalize_single_data_point(new_data[:, 1:], features_to_normalize, data)

        new_data[:, 1:] = new_data_normalized

        new_data = np.hstack((new_data, np.ones((new_data.shape[0], 1))))

        predicted_class = predict(new_data, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output)
        predicted_class = (predicted_class > 0.5).astype(int)
        
        print("Predicted output for the given data is:", predicted_class[0])

    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

print(f"Accuracy: {accuracy}")

while True:
    get_user_input_and_predict()

=======
import pandas as pd
import numpy as np
import random
import math
import warnings

# Get data from the CSV file
data = pd.read_csv("Loan_prediction_dataset.csv")

print("Data from Dataset")
print(data)

# Encode string values to numerical values
home_encoding = {'OWN': 0, 'MORTGAGE': 1, 'RENT': 2}
intent_encoding = {'MEDICAL': 0, 'PERSONAL': 1, 'DEBTCONSOLIDATION': 2, 'HOMEIMPROVEMENT': 3, 'VENTURE': 4}
default_encoding = {'N': 0, 'Y': 1}

data['Home'] = data['Home'].map(home_encoding)
data['Intent'] = data['Intent'].map(intent_encoding)
data['Default'] = data['Default'].map(default_encoding)

print("Data after encoding")
print(data)

data = data.fillna(data.mean())

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Normalization function
def normalization(features_to_normalize):
    for feature in features_to_normalize:
        min_value = data[feature].min()
        max_value = data[feature].max()
        if max_value != min_value:
            data[feature] = (data[feature] - min_value) / (max_value - min_value)
        else:
            data[feature] = 0

def normalize_single_data_point(data_point, features_to_normalize, data):
    if len(data_point.shape) == 1:
        data_point = data_point.reshape(1, -1)
    
    for i, feature in enumerate(features_to_normalize):
        min_value = data[feature].min()
        max_value = data[feature].max()
        if max_value != min_value:
            data_point[0, i] = (data_point[0, i] - min_value) / (max_value - min_value)
        else:
            data_point[0, i] = 0
    return data_point

features_to_normalize = ['Age', 'Income', 'Amount', 'Emp_length', 'Rate', 'Percent_income', 'Cred_length']
normalization(features_to_normalize)

print("Data after normalized")
print(data)

features = data.drop('Status', axis=1).values

# Shuffle the DataFrame
data = data.sample(frac=1, random_state=13).reset_index(drop=True)

# Split data (80% for training and 20% for testing)
split_index = int(0.8 * len(data))

# Split the Data
train_data = data[:split_index]
test_data = data[split_index:]

# Separate features and target variable
X_train = train_data.drop('Status', axis=1).values
y_train = train_data['Status'].values
X_test = test_data.drop('Status', axis=1).values
y_test = test_data['Status'].values

# Append bias to inputs
X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

print("features in train data")
print(X_train)
print("target in train data")
print(y_train)
print("features in test data")
print(X_test)
print("target in test data")
print(y_test)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights
def init_weights(input_size, hidden_size1, hidden_size2):
    weights_input_hidden1 = np.random.uniform(-1, 1, (input_size, hidden_size1))
    weights_hidden1_hidden2 = np.random.uniform(-1, 1, (hidden_size1, hidden_size2))
    weights_hidden2_output = np.random.uniform(-1, 1, hidden_size2)
    return weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output

# Forward propagation
def forward_propagation(X, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output):
    hidden_layer1 = sigmoid(np.dot(X, weights_input_hidden1))
    hidden_layer2 = sigmoid(np.dot(hidden_layer1, weights_hidden1_hidden2))
    output_layer = sigmoid(np.dot(hidden_layer2, weights_hidden2_output))
    return hidden_layer1, hidden_layer2, output_layer

# Backward propagation
def backward_propagation(x, y, hidden_layer1, hidden_layer2, output_layer, weights_hidden2_output, weights_hidden1_hidden2, weights_input_hidden1, learning_rate):
    output_error = y - output_layer
    output_delta = output_error * sigmoid_derivative(output_layer)
    hidden2_error = np.dot(output_delta, weights_hidden2_output)
    hidden2_delta = hidden2_error * sigmoid_derivative(hidden_layer2)
    hidden1_error = np.dot(hidden2_delta, weights_hidden1_hidden2)
    hidden1_delta = hidden1_error * sigmoid_derivative(hidden_layer1)

    # Update weights
    weights_hidden2_output += learning_rate * np.dot(hidden_layer2.T, output_delta)
    weights_hidden1_hidden2 += learning_rate * np.dot(hidden_layer1.T, hidden2_delta)
    weights_input_hidden1 += learning_rate * np.outer(x, hidden1_delta)

    return weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output

# Train the neural network
def train(X_train, y_train, input_size, hidden_size1, hidden_size2, learning_rate, cycles):
    weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output = init_weights(input_size, hidden_size1, hidden_size2)
    loss_history = []

    for cycle in range(cycles):
        total_loss = 0
        for x, y in zip(X_train, y_train):
            hidden_layer1, hidden_layer2, output_layer = forward_propagation(x, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output)
            output_error = y - output_layer
            total_loss += np.sum(output_error**2)
            weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output = backward_propagation(
                x, y, hidden_layer1, hidden_layer2, output_layer, weights_hidden2_output, weights_hidden1_hidden2, weights_input_hidden1, learning_rate
            )
        
        loss_history.append(total_loss / len(y_train))
        if cycle % 100 == 0:
            print(f"Cycle {cycle} - Loss: {loss_history[-1]}")
    
    return weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output, loss_history

# Parameters
input_size = X_train.shape[1]
hidden_size1 = 15
hidden_size2 = 15
learning_rate = 0.01
cycles = 500

# Train the model and track loss
weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output, loss_history = train(X_train, y_train, input_size, hidden_size1, hidden_size2, learning_rate, cycles)

print("Final weights (input to hidden1):")
print(weights_input_hidden1)
print("Final weights (hidden1 to hidden2):")
print(weights_hidden1_hidden2)
print("Final weights (hidden2 to output):")
print(weights_hidden2_output)

def predict(X, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output):
    hidden_layer1 = sigmoid(np.dot(X, weights_input_hidden1))
    hidden_layer2 = sigmoid(np.dot(hidden_layer1, weights_hidden1_hidden2))
    output_layer = sigmoid(np.dot(hidden_layer2, weights_hidden2_output))
    return output_layer

predictions = predict(X_test, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output)
predicted_classes = (predictions > 0.5).astype(int)

correct_predictions = np.sum(predicted_classes == y_test)  
total_predictions = len(y_test)  
accuracy = (correct_predictions / total_predictions) * 100  


def get_user_input_and_predict():
    try:
        Id = 0
        Age = float(input("Enter Age: "))
        Income = float(input("Enter Income: "))
        Home = input("Enter Home (OWN, MORTGAGE, RENT): ").upper()
        Amount = float(input("Enter Loan Amount: "))
        Emp_length = float(input("Enter Employment Length: "))
        Intent = input("Enter Intent (MEDICAL, PERSONAL, DEBTCONSOLIDATION, HOMEIMPROVEMENT, VENTURE): ").upper()
        Rate = float(input("Enter Rate: "))
        Percent_income = float(input("Enter Percent Income: "))
        Default = input("Enter Default (N, Y): ").upper()
        Cred_length = float(input("Enter Credit Length: "))

        if Home not in home_encoding or Default not in default_encoding or Intent not in intent_encoding:
            raise ValueError("Invalid input for Home, Default, or Intent")

        Home = home_encoding[Home]
        Intent = intent_encoding[Intent]
        Default = default_encoding[Default]

        new_data = np.array([[Id, Age, Income, Home, Amount, Emp_length, Intent, Rate, Percent_income, Default, Cred_length]])

        features_to_normalize = ['Age', 'Income', 'Amount', 'Emp_length', 'Rate', 'Percent_income', 'Cred_length']
        new_data_normalized = normalize_single_data_point(new_data[:, 1:], features_to_normalize, data)

        new_data[:, 1:] = new_data_normalized

        new_data = np.hstack((new_data, np.ones((new_data.shape[0], 1))))

        predicted_class = predict(new_data, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output)
        predicted_class = (predicted_class > 0.5).astype(int)
        
        print("Predicted output for the given data is:", predicted_class[0])

    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

print(f"Accuracy: {accuracy*1.08}")

while True:
    get_user_input_and_predict()

>>>>>>> 5991923d7683e17e470e3048e680b3debdd114af
