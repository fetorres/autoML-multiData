Author of original version of PARC autoML:  Raj Minhas, PARC
Contributors:  Dan Davies, PARC

#  Setting up your python environment	
This version is being tested using Python 3.5.
 It is being tested on Windows 64-bit and Linux Ubuntu 64-bit.  autoML.py should also work on MacOSx 64-bit
and 32-bit Windows, Linux, and MacOSx.  ARX, a Java program, has been modified to work with autoML for deprivatizing data with privacy concerns.  Unless the -s or --privatize_data flag is set to "none", Java needs to be installed for autoML.py to work.
The Python packages needed include NumPy, SciPy, cython, Pylab, and pyearth.  The installation of these
packages can be tricky on some systems.  Here is a process that has usually been working on Linux:<br>
<br>
Doing these in order is important-<br>
NumPy:    sudo apt-get install python3-numpy<br>
SciPy:    sudo apt-get install python3-scipy<br>
cython:   sudo pip3 install cython<br>
Pylab:    sudo pip3 install matplotlib  (pylab is part of matplotlib)<br>
Py-earth: go into the py-earth directory of your instance of this github repository and execute 'sudo python3 setup.py install'<br>
<br>
The above steps are currently being tested on MacOS.<br>
For Windows, I am using conda/anaconda.  Do the installations in the order above.  
<br>
The following packages get automatically installed if you are using Anaconda, but if not you may also need:<br>
    	  sudo pip3 install Pillow<br>
	  sudo pip3 install cov-core<br>
	  sudo pip3 install nose2<br>
	  sudo apt-get install python3-tk<br>
	  <br>
See Python_Windows_List_of_Packages.txt for the list of packages in a working Windows environment. (Not all packages are needed for autoML, but those that are have their version listed.)
<br>
<br>
# Executing autoML-multiData
This is an extension to autoML library built by Raj Minhas at PARC. 
autoML has features added for desensitizing data with privacy concerns.
A sample can be run by executing<br>
<br>
``` python
    python autoML.py  -i INPUT_FILE    (if run without the -i argument, it will run the time series data in data.csv)
```
<br>
    **usage:** autoML.py [-h] [-m MODEL_TYPE] [-i INPUT_FILE] [-file2 SECONDARY_FILE]
                     [-w SECONDARY_WEIGHTS] [-d DISTANCEFN] [-sprs SPARSITY]
                     [-r RADIUS] [-t MAX_TIME] [-n MAX_ITERATIONS] [-pca N_PCA]
                     [-e N_EXPERTS] [-s {none,manual,auto,manual_both,auto_both}]
                     [-f HIERARCHY_FOLDER]
    
    A simple interface to autoML (lite, multiData)
    
    optional arguments:<br>
      -h, --help            show this help message and exit
      -m MODEL_TYPE, --model_type MODEL_TYPE
                            choose classification(default), regression or
                            clustering
      -i INPUT_FILE, --input_file INPUT_FILE
                            primary input file to be analyzed (default=data.csv)
      -file2 SECONDARY_FILE, --secondary_file SECONDARY_FILE
                            optional secondary input file, triggers multi-dataset
                            analysis (default=None)
      -w SECONDARY_WEIGHTS, --secondary_weights SECONDARY_WEIGHTS
                            weights for features in secondary file, default=False
      -d DISTANCEFN, --distanceFn DISTANCEFN
                            choose L_1Norm(n), euclidean(n), L_infinityNorm(n),
                            distanceOnEarth(n), L_1Norm_cat(n), or
                            L_infinityNorm_cat(n), where n=1,2,3,... is the chosen
                            dimension for calculating distances. Default is
                            L_1Norm(1)
      -sprs SPARSITY, --sparsity SPARSITY
                            sparsity threshold for including records in secondary
                            input file
      -r RADIUS, --radius RADIUS
                            radius for cutoff of the distance function (default=1)
      -t MAX_TIME, --max_time MAX_TIME
                            maximum time in seconds for training all models. The
                            default value is 1440 seconds.
      -n MAX_ITERATIONS, --max_iterations MAX_ITERATIONS
                            max iterations for each individual model fit. The
                            default is 10 for clustering, 100 for classification
                            and 100 for regression.
      -pca N_PCA, --n_pca N_PCA
                            number of PCA components. (default is 2)
      -e N_EXPERTS, --n_experts N_EXPERTS
                            number of experts for Ensemble scoring. (default is 5)
      -s {none,manual,auto,manual_both,auto_both}, --privatize_data {none,manual,auto,manual_both,auto_both}
                            choose none, manual, or auto for privatization of the
                            data using ARX. For manual, an ARX window will launch.
                            For privatization of primary and secondary datasets,
                            choose manual_both or auto_both. (Default is 'manual')
      -f HIERARCHY_FOLDER, --hierarchy_folder HIERARCHY_FOLDER
                            folder containing hierarchy files for sensitive data,
                            if provided by the user. (Default is 'hierarchy')

It currently handles three types of models: classification, regression, and
clustering. If the model type is classification or regression then the last
column of the input data is assumed to be the dependent variable. Option to
add a second dataset and a distance function. The distance function is used to
assign elements of the second dataset to each row in the first dataset. A
cutoff radius is used for the selection, with default initial value of 1. The
-r option can be used to scale the distance function differently. 

##  Executing analysis with multiple datasets

A sample run could be with following command:

``` python
python3 autoML.py -m classification -i datasets/accident-landmarks-dataset/primary.data.csv -file2 datasets/accident-landmarks-dataset/secondary.data.csv -sprs 0.1 -r 1000 -d distanceOnEarth(2) -s none -t 300
```
	
## Subdirectories:
/example datasets - this subdirectory contains examples of how data from different 
datasets have been assembled into files autoML can use. 
