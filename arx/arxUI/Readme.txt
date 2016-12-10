This software is a modified version of the original ARX framework (version 3.4.1) at http://arx.deidentifier.org/. It is modified to be integrated with AutoML. The code for this software is available at https://githubenterprise.parc.com/abrito/fhwa-privacy/tree/master/arxUI. 

1. Compile and run the source code from Eclipse (tested on Eclipse version Mars.2 Release (4.5.2), java idk 1.8.0_92):
1) Download the source code from the arxUI folder at https://githubenterprise.parc.com/abrito/fhwa-privacy/tree/master/arxUI. 
2) In Eclipse, click "File - Import - Existing project into Workspace". Select the folder "arxUI" and click finish. The project should be imported into Eclipse workspace.
3) Click "Run - Run Configurations - Arguments". set the "Program Argument" as: [AutoML_folder] [input_data_path]. For example: data data/example.csv. 
4) Click Run. The software will start ARX and save the output files to [AutoML_folder] upon exit. 

2. Run the jar file from command line (tested on java are 1.8.0_92):
1) Download the jar file "arxUI.jar" at https://githubenterprise.parc.com/abrito/fhwa-privacy/tree/master/arxUI. If you want to use the example data, download the folder "data", and store it in the same directory as "arxUI.jar".
2) Go to the directory where "arxUI.jar" is stored and run: java -jar arx.jar [AutoML_folder] [input_data_path]. For example, run: java -jar arx.jar data data/example.csv. If you are using Mac OS, run: java –XstartOnFirstThread –jar arx.jar [AutoML_folder] [input_data_path]. 
3) The software will start ARX and save the output files to [AutoML_folder] upon exit. 
