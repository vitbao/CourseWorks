* The submitted supplementary files include the following:

1. integrator.py: this script integrate clinical data with genomic data sets and produced 4 different integrated data sets as described in the report. To run this script, use the following command provided that the input data files are available in an 'input folder'. The integrated data sets will be saved in the 'output folder'. 
	python integrator.py --Data=<input folder> --Output=<output folder> 
2. classifier.py: this script performs cross validation, train and test different classifiers on 4 different integrated data sets. It can be executed using the command below. It should be noted that the 'input folder' is the name of the folder that contains the integrated data sets. 'output file' is the name of the file that collects all the print statements, which includes cross validation results, predictive accuracy and confusion matrices. 
	python classifier.py --Data=<input folder> --out=<output file>
3. PREDICT_vs_Neuralnet.csv: this file contains the 100 test cases and the corresponding observations, the neural network's predictions and the predictions from PREDICT

Note:
* The following python packages are required in order to run the scripts, and the scripts are tested with python3.6: numpy, pandas, scikit-learn, keras, tensor-flow