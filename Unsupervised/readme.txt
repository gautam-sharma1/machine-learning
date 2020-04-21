# Midterm Group Project

This project makes use of K-means and EM algorithm to make predictions.

## Getting Started
This project consists of 3 programs written in python. First file is Kmeans.py that implements k means algorithm. Second file is GMM.py that implements EM algorithm. The third file is gmSuper.py that contains the class GMM implemented in GMM.py. There is no need to run this file but keep it in the same directory.

There is also one Matlab file called 'pre_post_proc.m' that generates statistics displayed in the report. It should be opened only to verify the statistics.

There is also a data2.csv provided that acts as the input data to the clustering algorithms. Unzip the folder and keep all contents in the same directory.

All the plots would be generated on their own. Each file has a variable NUMBER defined at the beginning. Change its value to change the number of clusters.

IMPORTANT : When figure window pops up, please close it to let the code proceed further. 
There is also one blank figure window generated. Please close it and ignore that. That does not represents any figure.

### Prerequisites
The program has the following dependencies:
Numpy
Pandas
Sklearn
Matplotlib
Seaborn
yellowbrick

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
$ pip install <package> #package=Numpy,Panda,Sklearn,Matplotlib,Seaborn,yellowbrick
```
It will get installed in the venv/ folder, and not conflict with other projects.

To leave the virtual environment run:
```
$ deactivate
```

## Running the tests
Open a terminal after activating the virtual environment and type:

```
python GMM.py # to run EM
python Kmeans.py # to run K means
```

Or you can run it through IDE like PyCharm after installing dependencies

### Break down into end to end tests
