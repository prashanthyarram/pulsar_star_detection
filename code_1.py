# imports for GUI
import tkinter as tk
from tkinter.ttk import *
from tkinter import *
# to handle tabular data
import numpy as np
import pandas as pd
# visualize the data
import seaborn as sns
import matplotlib.pyplot as plt
# Normalizing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# cross validation
from sklearn.model_selection import train_test_split
# Algorithms
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import syschk
import warnings
warnings.simplefilter("ignore")
# Data Acquisition
data = pd.read_csv(r"./pulsar_data_train.csv")
print(data)
# Data Analysis
data.info()
data.describe()
sns.set()
data.hist(figsize=(16, 10))
plt.tight_layout()
plt.show()
sns.countplot(y=data.target_class, data=data)
for index, value in enumerate(data["target_class"].value_counts()):
    plt.text(value, index, str(value))
plt.show()
plt.figure(figsize=(17, 17))
sns.pairplot(data=data, hue="target_class")
# plt.legend()
plt.tight_layout()
plt.show()
plt.figure(figsize=(17, 17))
sns.heatmap(data.corr(), annot=True)
plt.show()

# High Correlation Filter
removable_columns = set()
corr_mat = data.corr()
for i in range(len(corr_mat.columns)):
    for j in range(i):
        if abs(corr_mat.iloc[i, j]) > 0.9:
            colname = corr_mat.columns[i]
            removable_columns.add(colname)
print("Removable coloumns", removable_columns)
data.drop(columns=list(removable_columns), inplace=True)
print(data)

# Normalizing the values of the Dataset
scale = StandardScaler()
scale.fit(data.drop(columns=["target_class"]))
X = scale.transform(data.drop(columns=["target_class"]))
X = np.nan_to_num(X)
y = data["target_class"]
print(X)

# Data Partitioning
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Decision Tree
# instantiating the Algorithm
dtc = DecisionTreeClassifier()

# Training the model
dtc.fit(x_train, y_train)

# Testing the model
dtc_pred = dtc.predict(x_test)
# Evaluation
print("Accuracy Score: ", accuracy_score(y_test, dtc_pred)*100, "%")
print(classification_report(y_test, dtc_pred))
plt.figure(figsize=(5, 4))
plt.title("confusion matrix")
sns.heatmap(confusion_matrix(y_test, dtc_pred), annot=True, fmt="d")
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.show()

# RandomForestClassifier
# instantiating the Algorithm
rfc = RandomForestClassifier(random_state=0)
# Training the model
rfc.fit(x_train, y_train)
# Testing the model
y_pred = rfc.predict(x_test)

# Evaluation
print("Accuracy Score: ", accuracy_score(y_test, y_pred)*100, "%")
print(classification_report(y_test, y_pred))
plt.figure(figsize=(5, 4))
plt.title("confusion matrix")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
# plt.show()

# SVM
# instantiating the Algorithm
svm = SVC()
# Training the model
svm.fit(x_train, y_train)
# Testing the model
svm_pred = svm.predict(x_test)

# Evaluation
print("Accuracy Score: ", accuracy_score(y_test, svm_pred)*100, "%")
print(classification_report(y_test, svm_pred))
plt.figure(figsize=(5, 4))
plt.title("confusion matrix")
sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, fmt="d")
plt.xlabel("Predicted values")
plt.ylabel("Actual values")
plt.show()

class_names = ["Not a Pulsar", "Pulsar"]

# Initializing a GUI
root = tk.Tk()
root.geometry("550x300")
root.title("Pulsar Star Predection")

mean_i = tk.StringVar()
std_i = tk.StringVar()
ex_i = tk.StringVar()
mean_dm = tk.StringVar()
std_dm = tk.StringVar()
ex_dm = tk.StringVar()


def manual_pred(input_list):
    input_list = np.array(input_list)
    trans = scale.transform(input_list.reshape(1, -1))
    out = rfc.predict(trans)
    print(class_names[int(out[0])])
    newWindow = Toplevel(root)
    newWindow.title("Result")
    newWindow.geometry("200x200")
    g = class_names[int(out[0])]
    Label(newWindow,
          text=g).pack()


# Mean_of_the_integrated_profile = 140.562500
# Standard_deviation_integrated_profile = 55.683782
# Excess_kurtosis_integrated_profile = -0.234571
# Mean_DM_SNR_curve = 3.199833
# Standard_deviation_DM_SNR_curve = 19.110426
# Excess_kurtosis_DM_SNR_curve = `7.975532`


def submit():

    Mean_of_the_integrated_profile = float(mean_i.get())
    Standard_deviation_integrated_profile = float(std_i.get())
    Excess_kurtosis_integrated_profile = float(ex_i.get())
    Mean_DM_SNR_curve = float(mean_dm.get())
    Standard_deviation_DM_SNR_curve = float(std_dm.get())
    Excess_kurtosis_DM_SNR_curve = float(ex_dm.get())

    manual_pred([Mean_of_the_integrated_profile,
                 Standard_deviation_integrated_profile,
                 Excess_kurtosis_integrated_profile,
                 Mean_DM_SNR_curve,
                 Standard_deviation_DM_SNR_curve,
                 Excess_kurtosis_DM_SNR_curve])

    mean_i.set("")
    std_i.set("")
    ex_i.set("")
    mean_dm.set("")
    std_dm.set("")
    ex_dm.set("")


# Code for GUI
msg = tk.Label(
    root, text='Enter Float Values Only', font=('calibre', 10, 'bold',), foreground="red")
mean_i_label = tk.Label(
    root, text='Mean_of_the_integrated_profile', font=('calibre', 10, 'bold'))
mean_i_entry = tk.Entry(root, textvariable=mean_i,
                        font=('calibre', 10, 'normal'))

std_i_label = tk.Label(
    root, text='Standard_deviation_integrated_profile', font=('calibre', 10, 'bold'))
std_i_entry = tk.Entry(root, textvariable=std_i,
                       font=('calibre', 10, 'normal'))
ex_i_label = tk.Label(
    root, text='Excess_kurtosis_integrated_profile', font=('calibre', 10, 'bold'))
ex_i_entry = tk.Entry(root, textvariable=ex_i,
                      font=('calibre', 10, 'normal'))
mean_d_label = tk.Label(root, text='Mean_DM_SNR_curve',
                        font=('calibre', 10, 'bold'))
mean_d_entry = tk.Entry(root, textvariable=mean_dm,
                        font=('calibre', 10, 'normal'))

std_d_label = tk.Label(
    root, text='Standard_deviation_DM_SNR_curve', font=('calibre', 10, 'bold'))
std_d_entry = tk.Entry(root, textvariable=std_dm,
                       font=('calibre', 10, 'normal'))

ex_d_label = tk.Label(
    root, text='Excess_kurtosis_DM_SNR_curve', font=('calibre', 10, 'bold'))
ex_d_entry = tk.Entry(root, textvariable=ex_dm,
                      font=('calibre', 10, 'normal'))


sub_btn = tk.Button(root, text='Submit', command=submit)


mean_i_label.grid(row=0, column=0)
mean_i_entry.grid(row=0, column=1)
std_i_label.grid(row=1, column=0)
std_i_entry.grid(row=1, column=1)
ex_i_label.grid(row=2, column=0)
ex_i_entry.grid(row=2, column=1)
mean_d_label.grid(row=3, column=0)
mean_d_entry.grid(row=3, column=1)
std_d_label.grid(row=4, column=0)
std_d_entry.grid(row=4, column=1)
ex_d_label.grid(row=5, column=0)
ex_d_entry.grid(row=5, column=1)
msg.grid(row=6, column=1)
sub_btn.grid(row=8, column=1)

root.mainloop()
