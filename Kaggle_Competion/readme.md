# Midterm Project

Group project number 2. This project makes use of Linear Regression, Polynomial regression and MLP to make predictions.

## Getting Started
This project consists of 3 files written. 2 files are python codes and 1 file is the data file. Unzip the folder and keep all the files in one folder to make the program compile successfully. All the datasets Rainier_Wether.csv and Climbing_csv and the script competition_preprocessing.ipynb should be in the same folder
Run all the cells from the beginning and the 'final_competion.csv' will be created and saved as a csv  in the same folder 
that will be used further in training.

### Prerequisites
The program has the following dependencies:
Numpy
Pandas
Sklearn
Matplotlib

### Installing
Install the dependencies by making a virtual environment(recommended):

Open a terminal:
```
$ pip install virtualenv
$ cd my-project/ #name of the folder you want the virtual environment to be
$ virtualenv venv #venv is the name of the virtual environment. Can be anything.
```
These commands create a venv/ directory in your project where all dependencies are installed. You need to activate it first though (in every terminal instance where you are working on your project):
```
$ source venv/bin/activate
```
You should see a (venv) appear at the beginning of your terminal prompt indicating that you are working inside the virtualenv. Now when you install something like this:
```
$ pip install <package> #package=Numpy,Panda,Sklearn,Matplotlib
```
It will get installed in the venv/ folder, and not conflict with other projects.

To leave the virtual environment run:
```
$ deactivate
```

## Running the tests
Open a terminal after activating the virtual environment and type:

```
python 511project2a.py # to run Linear Regression
python 511project2b.py # to run MLP
python polynomial.py # to run Polynomial regression
```

Or you can run it through IDE like PyCharm after installing dependencies

### Break down into end to end tests
