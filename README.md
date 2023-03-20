# ML Basic Validation and Test Set v4
(complete outside of error messages)

##### Table of Contents

- [Description](#description)
- [How To Use](#how-to-use)
- [References](#references)

---

<p float="center">
    <img src="https://github.com/SDBranka/ML_Basic_Validation_and_Test_Set/blob/main/Resources/IMG/GUI%20startup.jpg" width=45% height= 315 alt="gui image"/>
    <img src="https://github.com/SDBranka/ML_Basic_Validation_and_Test_Set/blob/main/Resources/IMG/datascreenshot.jpg" width=45% height= 315 alt="graph image"/>
</p>

## Description

This app will will allow a user to validate their test model for a basic linear regression assessment and produce a graph based on the resulting data.


##### Controls

Once the app is properly set up and run, the user will be required to enter appropriate values for the following:
<ul>
    <li>the name of the training data file (without ".csv")</li>
    <li>the name of the testing data file (without ".csv")</li>
    <li>the desired learning rate for the model</li>
    <li>the desired scale factor for the label for the model<ul>
        <li>if the value of the data is too large this may beneficial</li></ul>
    </li>
    <li>the label to be predicted</li>
    <li>the feature</li>
    <li>and the desired size of the training set to pull from the training set data</li>
</ul>
After the user has entered values they may elect to run the test data in addition to the training/validation data by selecting the check box. When all values have been entered the user should click the "Run Program" button to produce a graph presenting the data curves.


##### Technologies

- Python
- tkinter
- NumPy
- Pandas
- Tensorflow
- matplotlib
- Visual Studio

---

## How To Use

<ul>
    <li>Download or clone this repository to your desktop</li>
    <li>A user should store their training data set and their test data set as .csv files in the data folder</li>
    <li>Run main.py in an appropriate Python environment</li>
</ul>

---

## References

##### Continuing Work on
- https://github.com/SDBranka/GG_ML_Crash_Course/blob/main/Validation_and_Test_Sets.py

\
[Back To The Top](#ml-basic-validation-and-test-set-v3)