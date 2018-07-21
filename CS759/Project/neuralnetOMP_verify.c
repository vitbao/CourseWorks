#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#define NUM_ATTR 60
#define NUM_INSTANCE 208

typedef struct {
	float attr_values[NUM_ATTR];
	char class_value[10];
	int index;
} Data;


typedef struct {
	float u[NUM_ATTR]; // weight vector
	float W[NUM_ATTR][NUM_ATTR]; // weight matrix
	float b[NUM_ATTR]; // bias vector
	float c; // bias scalar
} Network;

typedef struct {
	float h[NUM_ATTR]; // value of hidden units
	float o; // value of output unit
}NetworkValue;

typedef struct {
	float delta_u[NUM_ATTR]; // error in weight vector
	float delta_W[NUM_ATTR][NUM_ATTR]; // error in weight matrix
	float delta_b[NUM_ATTR]; // error in bias vector
	float delta_c; // error in output bias
} NetworkError;

void read_data(FILE *fp, Data *instance, char *labels[]); // read input data
void split_data(Data *instance, int folds, char *labels[], int *fold_instance_idx[], int fold_size[]); // split input data into folds
void initialize_network(Network *nw); // initialize network
NetworkValue feed_forward(Network *nw, float* x); // compute values of the network
void sigmoid_array(float *z); // sigmoid activation function
float sigmoid_scalar(float z); // sigmoid activation function
int binary_output(char *output, char *labels[]); // convert string output to binary number
NetworkError backpropagation_error(Data *instance, NetworkValue *nwValue, Network *nw, char *labels[]); // calculate errors of the network using back propagation algorithm
void update_network(Network *nw, NetworkError *nwError, float learning_rate); // Update network using computed network error
void print_data(Data *instance, int len); // utility function to print data instance
void print_network(Network *nw); // utility function to print network
void print_networkError(NetworkError *nwError); // utility function to print network error
void print_networkValue(NetworkValue *nwValue); // utility function to print network value
void train_network(Network *nw, Data *train_data, int epochs, int train_size, char * labels[], float learning_rate); //training network
void cross_validation(Data *instance, int folds, char *labels[], int epochs, float learning_rate, int num_threads); // k fold stratified cross validation
int test_network(Network *nw, Data *test_data, int test_size, char *labels[]); // test network

int main(int argc, char* argv[]){
	if (argc != 5)
	{
		printf("Error usage! Required 4 arguments: folds, epochs, learning_rate, num_threads\n");
		return 1;
	}
	// Open input file
	FILE *fp;
	fp = fopen("input.csv", "r");
	// Check if file is opened
	if (fp == NULL){ 
		printf("Error opening file: %s\n", strerror(errno));
		return 1;
	}
	// Get input parameters
	int num_folds = atoi(argv[1]);
	int epochs = atoi(argv[2]);
	float learning_rate = atof(argv[3]);
	int num_threads = atoi(argv[4]);

	// Read input data
	Data instance[NUM_INSTANCE];
	char *labels[2]; // character array to store class labels
	labels[0] = "";
	labels[1] = "";
	read_data(fp, instance, labels);
	fclose(fp);
	//print_data(instance, 2);
	// Cross Validation
	cross_validation(instance, num_folds, labels, epochs, learning_rate, num_threads);
	for (int i = 0; i < 2; i++)
		free(labels[i]);
	
}

void read_data(FILE *fp, Data *instance, char *labels[])
{
	char first_line[1024];
	char line[1024];
	char *token; // stores string after  line being splitted by comma 
	
	// Read the first line
	fgets(first_line, 1024, fp);
	// Read subsequent lines
	int i = 0;
	while (fgets(line, 1024, fp) != NULL){
		// Get first token
		token = strtok(line, ",");
		instance[i].attr_values[0] = atof(token);
		int j = 0;
		while(token != NULL){
			j++;
			token = strtok(NULL, ",");
			if (j < 60){
				instance[i].attr_values[j] = atof(token); 
 			}	
			else if (j == 60){ 	
				int len = strlen(token);
				for (int k = 0; k < len; k++)
				{
					if (token[k] == '\n' || token[k] == '\r')
					{
						token[k] = '\0'; // strip end-of-line and return characters	
					}
				}
				strcpy(instance[i].class_value, token);
				int isEmpty = strcmp(labels[1], "");
				if (i == 0){
					labels[0] = malloc(sizeof(char)*strlen(token));
					strcpy(labels[0], token);
				}
				else if (strcmp(labels[0], token) != 0 && isEmpty == 0){
				      	labels[1] = malloc(sizeof(char)*strlen(token));
					strcpy(labels[1], token);	
				}
			}
		}
		instance[i].index = i;
		i++;
	}
}


void cross_validation(Data *instance, int folds, char *labels[], int epochs, float learning_rate, int num_threads)
{
	int *fold_instance_idx[folds];
	int fold_size[folds];
	// Split data into k folds
	split_data(instance, folds, labels, fold_instance_idx, fold_size);
	int train_size;
	int test_size;
	Network nw;
	srand(0); // set seed
	int accuracy;
	// train the network using leave-one-out cross validation
	// if folds == 1, train set and test set are the same
	double total_time = 0.0;
	struct timeval start;
	struct timeval end;
	double start_OMP, end_OMP;
	for (int i = 0; i < 1; i++)
	{	
		if (folds == 1)
		{
			// Start timing
			gettimeofday(&start, NULL);
			train_size = NUM_INSTANCE;
			test_size = NUM_INSTANCE;
			Data train_set[train_size];
			Data test_set[test_size];
			for (int i = 0; i < NUM_INSTANCE; i++)
			{
				train_set[i] = instance[i];
				test_set[i] = instance[i];
			}
			// Initialize network
			initialize_network(&nw);
			// Train network
			train_network(&nw, train_set, epochs, train_size, labels, learning_rate);
			// Test network
			accuracy = test_network(&nw, test_set, test_size, labels);
			//printf("%d\n", accuracy);
			// End timing
			gettimeofday(&end, NULL);
			total_time += (end.tv_sec - start.tv_sec)*1000 + (double)(end.tv_usec - start.tv_usec)/1000;
		}
		else {
			// Start timing
			start_OMP = omp_get_wtime();
			omp_set_num_threads(num_threads);
			#pragma omp parallel for private(nw, test_size, train_size)
			for (int i = 0; i < folds; i ++)
			{
				test_size = fold_size[i];
				train_size = NUM_INSTANCE - test_size;
				Data train_set[train_size];
				Data test_set[test_size];
				int instance_idx;
				// Get test set
				for (int j = 0; j < test_size; j++)
				{
					instance_idx = fold_instance_idx[i][j];
					test_set[j] = instance[instance_idx];
				}
				int train_idx = 0;
				int train_subset_size;
				// Get train set
				for (int k = 0; k < folds; k++)
				{
					if (k != i)
					{
						train_subset_size = fold_size[k];
						for (int l = 0; l < train_subset_size; l++)
						{
							instance_idx = fold_instance_idx[k][l];
							train_set[train_idx] = instance[instance_idx];
							train_idx += 1;
						}		
					}	
				}
				// Initialize network
				initialize_network(&nw);
				// Train network
				train_network(&nw, train_set, epochs, train_size, labels, learning_rate);
				// Test network
				accuracy = test_network(&nw, test_set, test_size, labels);
				printf("Fold Index: %d%s%d\n",i, ", accuracy: ",  accuracy);
			}
			// End timing
			end_OMP = omp_get_wtime();
			total_time += (end_OMP - start_OMP)*1000; // in msec
		}
	}
	printf("%d%s%d%s%d%s%f\n", folds, ", ", epochs, ", ", num_threads, ", ",  total_time/1);
	//Deallocate memory
	for (int i = 0; i < folds; i++)
	{
		free(fold_instance_idx[i]);
	}
}
void split_data(Data *instance, int folds, char *labels[], int *fold_instance_idx[], int fold_sizes[])
{
	int num_neg = 0; // number of instances with 1st class value
	int num_pos = 0; // number of instances with 2nd class value
	
	int same; // compare instance class value with label	

	for (int i = 0; i < NUM_INSTANCE; i++)
	{
		same = strcmp(instance[i].class_value, labels[0]);
	       	if (same == 0)
			num_neg++;
		else
			num_pos++;
	}

	int neg_indices[num_neg]; // indices of negative instances in the original data
	int pos_indices[num_pos]; // indices of positive instances in the original data
	int neg_index = 0;
	int pos_index = 0;	
	for (int i = 0; i < NUM_INSTANCE; i++)
	{
		same = strcmp(instance[i].class_value, labels[0]);
		if (same == 0)
		{
			neg_indices[neg_index] = i;
			neg_index++;
		}
		else
		{
			pos_indices[pos_index] = i;
			pos_index++;
		}
	}
	
	//int fold_sizes[folds];
	int neg_fold_sizes[folds];
	int r = NUM_INSTANCE%folds;
	
	for (int i = 0; i < folds; i++)
	{
		if (i < r){
			fold_sizes[i] = NUM_INSTANCE/folds + 1;
			//fold_size[i] = fold_sizes[i];
		}
		else {
			fold_sizes[i] = NUM_INSTANCE/folds;
			//fold_size[i] = fold_sizes[i];
		}
	}
	
	r = num_neg%folds;
	for (int i = 0; i < folds; i++)
	{
		if (i< r)
			neg_fold_sizes[i] = num_neg/folds + 1;
		else
			neg_fold_sizes[i] = num_neg/folds;
	}
	
	// shuffle indices in neg_indices and pos_indices
	for (int i = num_neg - 1; i > 0; i--)
	{
		int c = rand()%(i+1);
		int temp = neg_indices[i];
		neg_indices[i] = neg_indices[c];
		neg_indices[c] = temp;
	}
	
	for (int i = num_pos - 1; i > 0; i--)
	{
		int c = rand()%(i+1);
		int temp = pos_indices[i];
		pos_indices[i] = pos_indices[c];
		pos_indices[c] = temp;
	}
       	
	// Populate fold_instance_idx array with indices of negative and positive instances
	// Each fold has different number of instances
	for (int i = 0; i < folds; i++)
	{
		int idx = 0;
		fold_instance_idx[i] = malloc(sizeof(int)*fold_sizes[i]);
		for (int j = 0; j < neg_fold_sizes[i]; j++)
		{
			fold_instance_idx[i][j] = neg_indices[idx + i*neg_fold_sizes[i]];
			idx++;
		}
		idx = 0;
		for (int j = neg_fold_sizes[i]; j< fold_sizes[i]; j++)
		{
			fold_instance_idx[i][j] = pos_indices[idx + i*(fold_sizes[i] - neg_fold_sizes[i])];
			idx++;
		}
		// shuffle indices in fold_instance_idx[i]
		for (int j = fold_sizes[i]-1; j > 0; j--)
		{
			int k = rand()%(j+1);
			int temp = fold_instance_idx[i][j];
			fold_instance_idx[i][j] = fold_instance_idx[i][k];
			fold_instance_idx[i][k] = temp;
		}
	}
}


void initialize_network(Network *nw)
{
	float random_num;
	for (int i = 0; i < NUM_ATTR; i++) 
	{
		//random_num = rand()/(float) RAND_MAX; // random number between 0 and 1
		random_num = 0.1; // used only to validate OMP result
		random_num = -1 + random_num * 2; // random number between -1 and 1
		nw->u[i] = random_num;
	}
	
	for (int i = 0; i < NUM_ATTR; i++)
	{
		for (int j = 0; j < NUM_ATTR; j++)
		{
			//random_num = rand()/(float)RAND_MAX;
			random_num = 0.2; // used only to validate OMP result
			random_num = -1 + random_num*2;
			nw->W[i][j] = random_num;
		}
	}
	for (int i = 0; i < NUM_ATTR; i++)
	{
		//random_num = rand()/(float)RAND_MAX;
		random_num = 0.3; // used only to validate OMP result
		random_num = -1 + random_num*2;
		nw->b[i] = random_num;
	}
	//random_num = rand()/(float)RAND_MAX; 
	random_num = 0.4; //used only to validate OMP result
	random_num = -1 + random_num*2; 
	nw->c = random_num;
}

void train_network(Network *nw, Data *instance, int epochs, int train_size, char *labels[], float learning_rate)
{
	for (int i = 0; i < epochs; i++)
	{
		NetworkValue nwValue;
		NetworkError nwError;
		for (int j = 0; j < train_size; j++)
		{
			nwValue = feed_forward(nw, instance[j].attr_values);
			nwError = backpropagation_error(&instance[j], &nwValue, nw, labels);
			update_network(nw, &nwError, learning_rate);
		}	
	}
}

void sigmoid_array(float *z)
{
	for (int i = 0; i < NUM_ATTR; i++)
		z[i] = 1/(1 + expf(-z[i]));	
	
}

float sigmoid_scalar(float z)
{
	return 1/(1 + expf(-z));
}

NetworkValue feed_forward(Network *nw, float* instance)
{
	NetworkValue nwValue;
	
	for (int i = 0; i < NUM_ATTR; i++)
	{	
		nwValue.h[i] = nw->b[i];
		for (int j = 0; j < NUM_ATTR; j++)
			nwValue.h[i] += nw->W[i][j] * instance[j];
	}
	sigmoid_array(nwValue.h);
	
	nwValue.o = nw->c;
	
	for (int i = 0; i < NUM_ATTR; i++)
		nwValue.o += nw->u[i]*nwValue.h[i];
      	nwValue.o = sigmoid_scalar(nwValue.o);	
	
	return nwValue;

}

int binary_output(char *output, char* labels[])
{	
	int same = strcmp(output, labels[0]);
	if (same == 0)
		return 0;
	else 
		return 1;
}

NetworkError backpropagation_error(Data *instance, NetworkValue *nwValue, Network *nw, char* labels[])
{
	NetworkError nwError;
	int output_value = binary_output(instance->class_value, labels);
	nwError.delta_c = (float)output_value - nwValue->o;
	for (int i = 0; i < NUM_ATTR; i++)
	{
		nwError.delta_b[i] = nwError.delta_c * (nw->u[i]);
		nwError.delta_u[i] = nwError.delta_c * (nwValue->h[i]);
		for (int j = 0; j < NUM_ATTR; j++)
		{
			nwError.delta_W[i][j] = nwError.delta_c * (nw->u[i]) * (nwValue->h[i]) * (1- (nwValue->h[i])) * (instance->attr_values[j]);
		}
	}
	return nwError;
}

void update_network(Network *nw, NetworkError *nwError, float learning_rate)
{
	nw->c = nw->c + learning_rate*(nwError->delta_c);
	for (int i = 0; i < NUM_ATTR; i++)
	{
		nw->b[i] += learning_rate*(nwError->delta_b[i]);
		nw->u[i] += learning_rate*(nwError->delta_u[i]);
		for (int j = 0; j < NUM_ATTR; j++)
			nw->W[i][j] += learning_rate*(nwError->delta_W[i][j]);
	}	
}

int test_network(Network *nw, Data *test_data, int test_size, char *labels[])
{
	int num_correct = 0;
	NetworkValue nwValue;
	char *predicted_label;
	char *actual_label;
	//printf("instance, actual label, predicted label: \n");
	for (int i = 0; i < test_size; i++)
	{
		nwValue = feed_forward(nw, test_data[i].attr_values);
		actual_label = malloc(sizeof(char)*strlen(test_data[i].class_value));
		strcpy(actual_label, test_data[i].class_value);
		if (nwValue.o < 0.5)
		{
			predicted_label = malloc(sizeof(char)*strlen(labels[0]));
			strcpy(predicted_label, labels[0]);
		}
		else
		{
			predicted_label = malloc(sizeof(char)*strlen(labels[1]));
			strcpy(predicted_label, labels[1]);
		}
		if (strcmp(predicted_label, actual_label) == 0)
			num_correct += 1;
		printf("instance index: %d%s%s%s%s\n", test_data[i].index, ", actual label: ", actual_label, ", predicted label: ", predicted_label);
		free(actual_label);
		free(predicted_label);
	}
	return num_correct;
}

void print_data(Data *instance, int len)
{
	for (int i = 0; i < len; i++)
	{	
		printf("%d%s", instance[i].index, ", ");
		for (int j = 0; j < NUM_ATTR; j++)
			printf("%f%s", instance[i].attr_values[j], ", ");
		printf("%s\n", instance[i].class_value);
	}
}

void print_network(Network *nw)
{
	printf("%f\n", nw->c);
	for (int i = 0; i < NUM_ATTR; i++)
		printf("%f%s", nw->b[i], ", ");
	printf("\n");
	for (int i = 0; i < NUM_ATTR; i++)
		printf("%f%s", nw->u[i], ", ");
	printf("\n");
	for (int i = 0; i < NUM_ATTR; i++)
	{
		for (int j = 0; j < NUM_ATTR; j++)
			printf("%f%s", nw->W[i][j], ", ");
		printf("\n");
	}
}

void print_networkError(NetworkError *nwError)
{
	printf("%f\n", nwError->delta_c);
	for (int i = 0; i < NUM_ATTR; i++)
		printf("%f%s", nwError->delta_b[i], ", ");
	printf("\n");
	for (int i = 0; i < NUM_ATTR; i++)
		printf("%f%s", nwError->delta_u[i], ", ");
	printf("\n");
	for (int i = 0; i < NUM_ATTR; i++)
	{
		for (int j = 0; j < NUM_ATTR; j++)
			printf("%f%s", nwError->delta_W[i][j], ", ");
		printf("\n");
	}
}
void print_networkValue(NetworkValue *nwValue)
{
	printf("%f\n", nwValue->o);
	for (int i = 0; i < NUM_ATTR; i++)
		printf("%f%s", nwValue->h[i], ", ");
	printf("\n");
}

