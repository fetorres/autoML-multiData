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

## Testing autoML

We use the nose2 testing framework.  See http://nose2.readthedocs.io/en/latest/getting_started.html.  The command to install nose2 was shown above.<br>
nose2 looks for packages, files and methods whose names begin with "test".  There is a file in autoML-multiData
named "test_autoML.py".  It loads autoML.py and runs it with the "-x" (run regression test) flag set.  To invoke the tests that include coverage and pylint, cd to /mnt/disks/disk-1/autoML-multiData and type the following.  Note that PYTHONPATH need only be initialized once<br>
``` sh
export PYTHONPATH=/mnt/disks/disk-1/autoML-multiData:$PYTHONPATH
nose2 --with-coverage
```
# Setting up jenkins

These are the steps I followed to install and initialize jenkins.  After doing this, I discovered the Jenkins Handbook on https://jenkins.io/user-handbook.pdf which covered some of the same ground.  It also has some big holes where no one has volunteered to write the documentation.  http://www.tutorialspoint.com/jenkins/ and http://www.vogella.com/tutorials/Jenkins/article.html are also nice.  

1. Start by reading https://jenkins.io/download/  Click on the link entitled "Installing Jenkins on Ubuntu", which takes you to 'https://wiki.jenkins-ci.org/display/JENKINS/Installing+Jenkins+on+Ubuntu'.

2. If they're not already present on your soon-to-be jenkins server, install the java jdk and jre.  The default versions of these are from OpenJDK.  In the distant past, OpenJDK had problems with java constructs that worked fine in Oracle Java, so I normally just install Oracle Java.  See the instructions for updating your repository list and installing java 8 on https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-get-on-ubuntu-16-04.

3. Back on https://wiki.jenkins-ci.org/display/JENKINS/Installing+Jenkins+on+Ubuntu, follow the instructions labelled "Installation".  I skipped the rest of the instructions, though they might be useful sometime.

4. Open a browser (I used chromium-browser) on the target jenkins server an use it to view "localhost:8080"  Let me emphasize that - you need to log into the server on which jenkins will run.  You'll probably use SSH for this.  SSH may want you to use private keys.  On that server, open a browser.  You do not was a browser that opens on the desktop of the machine that you're using to connect to the soon-to-be-running-jenkins server.  For example, ddlinux.parc.xerox.com is my office Linux box.  It is receiving my keystrokes and using SSH to send them to a Google Cloud server that has an exterior IP Address and is named "automl-test-v1".  You want the browser to be invoked from automl-test-v1.

5. It starts by telling you to copy the special super-secret administrator's key from a file into the initializer.  Do it.

6. Let it load its favorite plugins.

7. Fill in the info that specifies you as the first administrator.

8. You get a screen with a cheery button that says "Start Using Jenkins".  Click it.

9. I tried to use GitHub authentication for Jenkins, but couldn't get it to work.  When it was set up and you logged off of jenkins, you couldn't log back on.  There was evidence that jenkins asked github for data (github put up messages on the console asking if handing the jenkins was ok) but I never saw an indication that jenkins did anything useful after that.  So, we now let jenkins keep its own user database.  We also use the plain old git plugin, not the fancy github plugin.

10. Press "Manage Jenkins", then "Configure System" at the top of the column of icons.  Scroll down and set the "Jenkins URL" to a valid URL instead of "localhost".  Since the pretty name that google cloud gave to our server means nothing outside of google cloud, I think you have to use the naked IP Address.  Also scroll to the "Git" section an fill in the user.name and user.email fields.  I *think* that you're specifying name of the jenkins server, which is "jenkins", not your name.  I *think* that it's ok to set the suer.email value to your email address.  Remember to press the "Save" button!

* Press Manage Jenkins", then "Configure Global Security"  Under "Access Control" and "Security Realm", press "Jenkins' own user database" and "Allow users to sign up" below it.

* To add users, press the "Manage Jenkins", then "Manage Users" icon.  Click on the "Create User" icon on the far left of the screen and add whatever info you think is appropriate.

* To give your users permissions to do things, press "Manage Jenkins", then "Configure Global Security".  Under "Access Control" and "Authorization", click "Matrix-based security", type in your user's name in the "user/group to add" box, add it, and assign whatever permissions you like.  Press Save!

* We're going to want to access a private gitub respository, so we'll need a public/private key pair.  We need to use ssh-keygen to do that, but that requires sudo.  Here's where things get a bit squirrelly.  If you used a key pair to open the SSH connection to your jenkins server machine, you may not have a password.  Either sudo or ssh-keygen wants a password.  So, give yourself one using "sudo passwd your.username

* Once you have a password, use "sudo -u jenkins ssh-keygen to get a public/private pair.  I accepted all the defaults.

* You can fetch more plugins at this point if you wish.  Click "Manage Jenkins", the "Manaage Plugins".  The "Installed" tab tells you which plugins you have.  The "Available" tab lists plugins that you might want.  Once you install the plugings, it seems to be a good idea to restart Jenkins.  There are several ways to do this.  One is to tack "/restart" onto the end of the URL that you're using to contact jenkins.  For example, "http://localhost:8080/restart"

* to create a test job, I basically followed the section labelled "Setting up a Jenkins job" from http://www.vogella.com/tutorials/Jenkins/article.html.  I had git fetch data from my repo.  I used the 15-minute polling scheduleing scheme.  I had it invoke a script ("nose2 --with-coverage") and turned off the building step since python doesn't need to be compiled or linked.  With those changes, it seems to be working!






	
