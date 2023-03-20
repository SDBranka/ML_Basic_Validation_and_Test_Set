import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import tkinter as tk


# ---------------------------- pandas Setup ------------------------------- #
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format


# ---------------------------- Variables ------------------------------- #
# for GUI
LABEL_FONT = ("Arial", 12, "normal")


# ---------------------------- Functions ------------------------------- #
# for program tests
# Define the model's topography
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
def plot_the_loss_curve(run_test_data, epochs, mae_training, mae_validation):
    # Plot a curve of loss vs. epoch.
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
    # plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
    if run_test_data == 1:
        plt.plot(epochs[1:], mae_validation[1:], label="Test Data Loss")
    else:
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


# for GUI
# def close_warning():
#     err_screen.destroy()


def error_msg(warning_msg):
    global err_screen
    err_screen = tk.Toplevel(window)
    err_screen.title("Warning")
    err_screen.config(padx=27, pady=27)
    err_label = tk.Label(err_screen, 
                        text=f"{warning_msg}",
                        fg = "red"
    )
    err_label.grid(row = 0, column = 0, columnspan = 3)
    # warning_button = tk.Button(err_screen,
    #                             text = "Ok",
    #                             command = close_warning,
    #                             fg = "white",
    #                             bg = "red",
    #                             pady=12
    # )
    # warning_button.grid(row = 2, column = 1)


def run_program_button_clicked():
    # retrieve values from the entry fields
    training_data = training_data_input.get()
    testing_data = testing_data_input.get()
    my_feature = my_feature_input.get()
    my_label = my_label_input.get()
    scale_factor = scale_factor_input.get()
    epochs = epochs_input.get()
    validation_split = validation_split_input.get()
    learning_rate = learning_rate_input.get()
    batch_size = batch_size_input.get()
    training_set_size = training_set_size_input.get()
    run_test_data = run_test_data_state.get()

    # check validation score
    check_score = 0
    # value validations
    # are there no empty entries (excluding test_data, handled later)
    if len(training_data) == 0 or len(my_feature) == 0 or len(my_label) == 0 or len(scale_factor) == 0 or len(epochs) == 0 or len(validation_split) == 0 or len(learning_rate) == 0 or len(batch_size) == 0 or len(training_set_size) == 0:
            error_msg("No field may \n be left empty")
    else:
        check_score += 1

        while check_score < 7:
            # are the number types appropriate
            try:
                scale_factor = float(scale_factor)  
                validation_split = float(validation_split)  
                learning_rate = float(learning_rate) 
                check_score += 1
            except ValueError:
                error_msg('Please make sure that scale factor, validation split, \n and learning rate are all valid decimal numbers')
            try:
                epochs = int(epochs) 
                batch_size = int(batch_size) 
                training_set_size = int(training_set_size) 
                run_test_data = int(run_test_data)
                check_score += 1
            except ValueError:
                error_msg('Please make sure that epochs, batch size, \n and training set size are all valid whole numbers')
            
            # are all numbers >= 0
            if scale_factor <= 0 or epochs <= 0 or validation_split <= 0 or learning_rate <= 0 or batch_size <= 0 or training_set_size <= 0:
                error_msg('No field may be \n 0 or negative')
            else:
                check_score += 1

            # does the training file exist
            try:
                train_df = pd.read_csv(f"data/{training_data}.csv")
                check_score += 1
            except:
                error_msg('That training data \n file does not exist')

            # validate the feature
            try:
                train_df[my_feature] == True
                check_score += 1
            except:
                error_msg('That feature column does not \n exist in the training data set file')

            # validate and scale label
            try:
                train_df[my_label] == True
                train_df[my_label] /= scale_factor
                check_score += 1
            except:
                error_msg('That label column does not \n exist in the training data set file')

            
        test_df = None
        # if the user elects to run the test data
        if run_test_data == 1:
            while check_score < 11:
                # does the testing file exist
                if len(testing_data) == 0:
                    error_msg('You must enter a \n test data file')
                else:
                    check_score += 1

                try:
                    test_df = pd.read_csv(f"data/{testing_data}.csv")
                    check_score += 1
                except:
                    error_msg('That testing data \n file does not exist')

                try:
                    test_df[my_feature] == True
                    check_score += 1
                except:
                    error_msg('That feature column does not \n exist in the testing data set file')

                try:
                    # Scale the test set's label
                    test_df[my_label] /= scale_factor
                    check_score += 1
                except:
                    error_msg('That label column does not \n exist in the testing data set file')


        # run the program
        train_df.head(n=training_set_size)

        my_model = build_model(learning_rate)

        shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))

        epochs, rmse, history = train_model(my_model, shuffled_train_df, my_feature, 
                                            my_label, epochs, batch_size, 
                                            validation_split)

        plot_the_loss_curve(run_test_data, epochs, history["root_mean_squared_error"], 
                        history["val_root_mean_squared_error"])
        
        if run_test_data == 1:
            # Use the Test Dataset to Evaluate Model's Performance
            # The test set usually acts as the ultimate judge of a 
            # model's quality. The test set can serve as an impartial 
            # judge because its examples haven't been used in training 
            # the model. 
            x_test = test_df[my_feature]
            y_test = test_df[my_label]

            results = my_model.evaluate(x_test, y_test, batch_size=batch_size)


# ---------------------------- UI SETUP ------------------------------- #
# build window
window = tk.Tk()
window.title("Validation and Test Sets")
# add padding between the window frame and the elements
window.config(padx=27, pady=27)


# column 0
# training data label and entry
training_data_label = tk.Label(text = "Enter the name of your \n training data file: ",
                                font = LABEL_FONT
)
training_data_label.grid(row = 1, column = 0, columnspan = 3)

training_data_input = tk.Entry(width = 72)
# place the tying cursor inside the field
training_data_input.focus()
training_data_input.grid(row = 2, column = 0, columnspan = 3)


# testing data label and entry
testing_data_label = tk.Label(text = "Enter the name of your \n testing data file: ",
                                font = LABEL_FONT
)
testing_data_label.grid(row = 3, column = 0, columnspan = 3)

testing_data_input = tk.Entry(width = 72)
testing_data_input.grid(row = 4, column = 0, columnspan = 3)


# my feature label and entry
my_feature_label = tk.Label(text = "Enter the name \n of your feature: ",
                                font = LABEL_FONT
)
my_feature_label.grid(row = 5, column = 0, columnspan = 3)

my_feature_input = tk.Entry(width = 72)
my_feature_input.grid(row = 6, column = 0, columnspan = 3)


# my label label and entry
my_label_label = tk.Label(text = "Enter the name of your \n label to be predicted: ",
                                font = LABEL_FONT
)
my_label_label.grid(row = 7, column = 0, columnspan = 3)

my_label_input = tk.Entry(width = 72)
my_label_input.grid(row = 8, column = 0, columnspan = 3)


# scale factor label and entry
scale_factor_label = tk.Label(text = "Enter your desired scale factor \n (must be greater than 0.0): ",
                                font = LABEL_FONT
)
scale_factor_label.grid(row = 9, column = 0)

scale_factor_input = tk.Entry(width = 36)
# prepopulate field with the most common entry
scale_factor_input.insert(tk.END, 1000.0)
scale_factor_input.grid(row = 10, column = 0)


# epochs label and entry
epochs_label = tk.Label(text = "Enter your desired \n number of epochs:",
                                font = LABEL_FONT
)
epochs_label.grid(row = 11, column = 0)

epochs_input = tk.Entry(width = 36)
# prepopulate field with the most common entry
epochs_input.insert(tk.END, 30)
epochs_input.grid(row = 12, column = 0)


# validation split label and entry
validation_split_label = tk.Label(text = "Enter your desired validation split \n (must be 0.1-1.0):",
                                font = LABEL_FONT
)
validation_split_label.grid(row = 13, column = 0)

validation_split_input = tk.Entry(width = 36)
# prepopulate field with the most common entry
validation_split_input.insert(tk.END, 0.2)
validation_split_input.grid(row = 14, column = 0)


# column 1


# column 2
# learning rate label and entry
learning_rate_label = tk.Label(text = "Enter your desired learning rate \n (must be greater than 0.0): ",
                                font = LABEL_FONT
)
learning_rate_label.grid(row = 9, column = 2)

learning_rate_input = tk.Entry(width = 36)
# prepopulate field with the most common entry
learning_rate_input.insert(tk.END, 0.08)
learning_rate_input.grid(row = 10, column = 2)


# batch size label and entry
batch_size_label = tk.Label(text = "Enter your desired \n batch size:",
                                font = LABEL_FONT
)
batch_size_label.grid(row = 11, column = 2)

batch_size_input = tk.Entry(width = 36)
# prepopulate field with the most common entry
batch_size_input.insert(tk.END, 100)
batch_size_input.grid(row = 12, column = 2)


# training set size label and entry
training_set_size_label = tk.Label(text = "Enter your desired \n training set size:",
                                font = LABEL_FONT
)
training_set_size_label.grid(row = 13, column = 2)

training_set_size_input = tk.Entry(width = 36)
# prepopulate field with the most common entry
training_set_size_input.insert(tk.END, 10000)
training_set_size_input.grid(row = 14, column = 2)


# column 3
# Check to run test data
run_test_data_state = tk.IntVar()
run_test_data_checkbutton = tk.Checkbutton(text="Run test data", 
                            variable = run_test_data_state, 
)
run_test_data_checkbutton.grid(row = 2, column = 3)


# run program button
run_program_button = tk.Button(text = "Run analysis",
                        command = run_program_button_clicked,
                        fg = "white",
                        bg = "blue"
)
run_program_button.grid(row = 12, column = 3)




# to maintain window during use
window.mainloop()















