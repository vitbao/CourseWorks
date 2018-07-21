## Project Title
Speed up neural network training using parallel computing with OpenMP

## List of files
* neuralnet.c
* neuralnetOMP.c
* neuralnetOMP1.c
* neuralnet_verify.c
* neuralnetOMP_verify.c
* neuralnetOMP1_verify.c
* Makefile
* input.csv
* gprofAnalysis.txt
* neuralnet_verify.sbatch
* OMP_verify.sbatch
* OMP1_verify.sbatch

## Source codes' description
* neuralnet.c:
	* Sequential neural network training algorithm with initial network parameters randomly assigned. 
	* Execution time is measured 10 times.
	* Used for scaling analysis
	* Referred to as algorithm M1 in the report
* neuralnetOMP.c:
	* Parallel neural network training algorithm with initial network parameters randomly assigned. 
	* Execution time is measured 10 times.
	* Used for scaling analysis
	* Referred to as algorithm M2 in the report
* neuralnetOMP1.c:
	* Parallel neural network training algorithm with initial network parameters randomly assigned. 
	* Execution time is measured 10 times.
	* Used for scaling analysis
	* Referred to as algorithm M3 in the report
* neuralnet_verify.c: Same as M1 with some modifications:
	* Initial network parameters fixed to constant values. 
	* Execution time is measured only once.
	* Outputs is used as reference to verify OMP algorithms' outputs. 
* neuralnetOMP_verify.c: Same as M2 with some modifications:
	* Initial network parameters fixed to constant values. 
	* Execution time is measured only once. 
	* Outputs is compared with reference outputs to verify algorithm's correctness.
* neuralnetOMP1_verify.c: Same as M3 with some modifications:
	* Initial network parameters fixed to constant values. 
	* Execution time is measured only once.
	* Outputs is compared with reference outputs to verify algorithm's correctness.

## Other files' description
* input.csv: Data used to train the neural network
* gprofAnalysis.txt: Output from running grof profiling tool for neuralnet_verify program with 5 fold, 25 epochs and learning rate = 0.1
* neuralnet_verify.sbatch: Script to produces reference (sequential) outputs
* OMP_verify.sbatch: Script to produces parallel outputs	
* OMP1_verify.sbatch: Script to Produces parallel outputs

### To produce executable programs
```
make
```
### To verify the correctness of parallel algorithms

```
sbatch neuralnet_verify.sbatch	
sbatch OMP_verify.sbatch
sbatch OMP1_verify.sbatch
```
* Note: the order of outputs from parallel programs might be different from reference outputs. However, the outputs should be the same for instances with the same index	

### To run sequential algorithm
```
./neuralnet num_folds epochs learning_rate 
```
* The outputs are: num_folds, epochs, average execution time
* Example: ./neuralnet 5 25 0.1


### To run parallel algorithms
```
./neuralnetOMP num_folds epochs learning_rate num_threads
./neuralnetOMP1 num_folds epochs learning_rate num_threads
```
* The outputs are num_folds, epochs, num_threads, average execution time
* Example: ./neuralnetOMP 5 25 0.1 5
