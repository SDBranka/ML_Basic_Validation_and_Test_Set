import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt


# ---------------------------- pandas Setup ------------------------------- #
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format


# ---------------------------- Variables ------------------------------- #
# training_data = "california_housing_train"
training_data = "."
# test_data = "california_housing_test"
testing_data = "."
# scale_factor = 1000.0
scale_factor = 0
my_label = "."
my_feature = "."
# The following variables are the hyperparameters.
# learning_rate = 0.08
learning_rate = 0
epochs = 30
batch_size = 100
# Split the original training set into a reduced training 
# set and a validation set. 
validation_split = 0.2
# Identify the feature and the label.
# my_feature = "median_income"    # the median income on a specific city block.
# my_label = "median_house_value" # the median house value on a specific city block.
# That is, you're going to create a model that predicts 
# house value based solely on the neighborhood's median income.  
# Set training set size
# training_set_size = 10000
training_set_size = 0


# ---------------------------- Functions ------------------------------- #
#Define the model's topography
def build_model(my_learning_rate):
    # Create and compile a simple linear regression model.
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Add one linear layer to the model to yield a simple 
    # linear regressor.
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    # Compile the model topography into code that TensorFlow 
    # can efficiently execute. 
    # Configure training to minimize the model's mean squared error. 
    model.compile(optimizer=tf.keras.optimizers.experimental.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model  


# Train the model, outputting not only the loss value for the training 
# set but also the loss value for the validation set
def train_model(model, df, feature, label, my_epochs, 
                my_batch_size=None, my_validation_split=0.1):
    # Feed a dataset into the model in order to train it.
    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=my_batch_size,
                        epochs=my_epochs,
                        validation_split=my_validation_split)

    # Gather the model's trained weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the 
    # rest of history.
    epochs = history.epoch
    
    # Isolate the root mean squared error for each epoch.
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return epochs, rmse, history.history   


# Define plotting function
# The plot_the_loss_curve function plots loss vs. epochs 
# for both the training set and the validation set.
def plot_the_loss_curve(epochs, mae_training, mae_validation):
    # Plot a curve of loss vs. epoch.
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
    plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
    plt.legend()

    # We're not going to plot the first epoch, since the loss 
    # on the first epoch is often substantially greater than 
    # the loss for other epochs.
    merged_mae_lists = mae_training[1:] + mae_validation[1:]
    highest_loss = max(merged_mae_lists)
    lowest_loss = min(merged_mae_lists)
    delta = highest_loss - lowest_loss
    print(delta)

    top_of_y_axis = highest_loss + (delta * 0.05)
    bottom_of_y_axis = lowest_loss - (delta * 0.05)

    plt.ylim([bottom_of_y_axis, top_of_y_axis])
    plt.show()  


# ---------------------------- Establish Data and Variables ------------------------------- #
# ask the user to enter the name of the file to be used for training data
# continue to ask until the user enters a usable file
while training_data == ".":
    training_data = input("enter the name of your training data file: ")
    # print(f"training_data: {training_data}")
    
    # backdoor for testing
    if training_data == "a":
        break    
    
    try:
        train_df = pd.read_csv(f"../../data/{training_data}.csv")
    except:
        print("that training file does not exist")
        training_data = "."

# ask the user to enter the name of the file to be used for testing data
# continue to ask until the user enters a usable file
while testing_data == ".":
    testing_data = input("enter the name of your testing data file: ")
    # print(f"testing_data: {testing_data}")
    
    # backdoor for testing
    if testing_data == "a":
        break    
    
    try:
        test_df = pd.read_csv(f"../../data/{testing_data}.csv")
    except:
        print("that testing file does not exist")
        testing_data = "."

# ask the user to set the learning rate for the data
# continue to ask until the user enters a usable value
while learning_rate == 0:
    try:
        learning_rate = float(input("Enter your desired learning rate (must be greater than 0.0): "))
        if learning_rate <= 0:  # if not a positive int print message and ask for input again
            print("Sorry, input must be a positive integer and greater than 0, try again")
            continue
        break  

    except ValueError:
        print("Not an integer! Try again.")

# ask the user to set the scale factor for the data
# continue to ask until the user enters a usable value
while scale_factor == 0:
    try:
        scale_factor = float(input("Enter your desired scale factor (must be greater than 0.0): "))
        if scale_factor <= 0:  # if not a positive int print message and ask for input again
            print("Sorry, input must be a positive integer and greater than 0, try again")
            continue
        break       

    except ValueError:
        print("Not an integer! Try again.")

# ask the user to set and scale label
# continue to ask until the user enters an existing column
while my_label == ".":
    my_label = input("Enter the name of your label to be predicted: ")
    
    # backdoor for testing
    if my_label == "a":
        break    
    
    try:
        # Scale the training set's label.
        train_df[my_label] /= scale_factor 
        # Scale the test set's label
        test_df[my_label] /= scale_factor

    except:
        print("That column does not exist")
        my_label = "."

# ask the user to set the feature
# continue to ask until the user enters an existing column
while my_feature == ".":
    my_feature = input("Enter the name of your feature: ")
    
    # backdoor for testing
    if my_feature == "a":
        break    
    
    try:
        train_df[my_feature] == True

    except:
        print("That column does not exist")
        my_feature = "."

# ask the user to set training set size
# continue to ask until the user enters a usable value
while training_set_size == 0:
    try:
        training_set_size = int(input("Enter your desired sample size (must be greater than 0): "))    
        if scale_factor <= 0:  # if not a positive int print message and ask for input again
            print("Sorry, input must be a positive integer and greater than 0, try again")
            continue
        break       

    except ValueError:
        print("Not an integer! Try again.")


# ---------------------------- Program Cycle ------------------------------- #
# Invoke the functions to build and train the model.
train_df.head(n=training_set_size)

my_model = build_model(learning_rate)

shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))

epochs, rmse, history = train_model(my_model, shuffled_train_df, my_feature, 
                                    my_label, epochs, batch_size, 
                                    validation_split)

plot_the_loss_curve(epochs, history["root_mean_squared_error"], 
                    history["val_root_mean_squared_error"])


# Use the Test Dataset to Evaluate Your 
# Model's Performance
# The test set usually acts as the ultimate judge of a 
# model's quality. The test set can serve as an impartial 
# judge because its examples haven't been used in training 
# the model. 
x_test = test_df[my_feature]
y_test = test_df[my_label]

results = my_model.evaluate(x_test, y_test, batch_size=batch_size)



