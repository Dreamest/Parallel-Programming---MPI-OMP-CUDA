#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper.h"

#define PATH_FILE "C:\\inputFile.txt"
#define ROOT 0
#define PROCCESSES 2

///----Function declerations---///
extern cudaError_t calculateWithGPU(int *arr, int *results, unsigned int size);
double f(int i);
int IterativeCalculation(int* arr, int size); //For iterative solution - Only for checking.
int calculateWithOMP(int* myArr,int mySize);
int calculateWithCUDA(int* myArr,int mySize);
int handleLeftover(int* arr, int size);

///----Main function---///
int main(int argc, char* argv[])
{
	int *arr, size, *myArr, mySize, totalSum = 0, mySum = 0;
	int myId, numOfProcesses, halfSize;
	int iterativeSum = 0; //For iterative solution - Only for checking.
	int i;
	FILE* fp;
	double t0, t1; 

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myId);
	MPI_Comm_size(MPI_COMM_WORLD, &numOfProcesses);

	if (numOfProcesses != PROCCESSES) //Number of proccesses should be 2
	{
		printf("\nInvalid number of processes: %d instead of %d\n", numOfProcesses, PROCCESSES);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if (myId == ROOT) {
		//Read array from file
		fopen_s(&fp, PATH_FILE, "r");
		if (!fp) //if file doesn't open - abort.
		{
			printf("File could not open.\n");
			MPI_Abort(MPI_COMM_WORLD, 2);
		}
		fscanf_s(fp, "%d", &size);
		arr = (int*)calloc(size, sizeof(int));
		if (!arr)
			MPI_Abort(MPI_COMM_WORLD, 2);

		for (i = 0; i < size; i++)
			fscanf_s(fp, "%d", &arr[i]);
		fclose(fp);

		t0 = MPI_Wtime(); //Start measuring time for iterative solution.
		iterativeSum = IterativeCalculation(arr, size); // Don't need
		t1 = MPI_Wtime(); //Stop measuring time for iterative solution.
		printf("\n(Iterative) The total number of positive results is: %d \nIt took %lf seconds\n", iterativeSum, (t1-t0)); //For iterative solution - Only for checking.
		
		t0 = MPI_Wtime();//Start measuring time for parallel solution.
		mySize = size / PROCCESSES; //Size of the array each process will get
	}

	MPI_Bcast(&mySize, 1, MPI_INT, ROOT, MPI_COMM_WORLD); //Broadcast mySize to all processes

	myArr = (int*)calloc(mySize, sizeof(int)); 
	if (!myArr)
		MPI_Abort(MPI_COMM_WORLD, 2);

	MPI_Scatter(arr, mySize, MPI_INT, myArr, mySize, MPI_INT, ROOT, MPI_COMM_WORLD); //Scatter arr to all processes

	halfSize = mySize / 2; // A quarter of the original array.

	//In each process:
	mySum += calculateWithOMP(myArr, halfSize); //first half with OMP
	mySum += calculateWithCUDA(myArr + halfSize, halfSize + mySize%2); //second half with CUDA

	//If the array is uneven, leftover is handled by root.
	if (myId == ROOT && size % 2 != 0)
		mySum += handleLeftover(arr, size);

	MPI_Reduce(&mySum,&totalSum,1,MPI_INT,MPI_SUM,ROOT, MPI_COMM_WORLD); //reduce mySum from each process to totalSum

	if (myId == ROOT) {
		t1 = MPI_Wtime(); //Stop measuring time for parallel solution.
		printf("\n(Parallel) The total number of positive results is: %d \nIt took %lf seconds\n", totalSum, (t1-t0));
		free(arr);
	}

	free(myArr);
	MPI_Finalize();
	return 0;
}

///----Other functions---///

//Runs function f on all the array without using parallelization
int IterativeCalculation(int* arr, int size)
{
	int i;
	int counter = 0;
	for (i = 0; i < size; i++)
		if (f(arr[i]) > 0) 
		{
			counter++;
		}
	return counter;
}

//calculating the last element, if need to.
int handleLeftover(int* arr, int size)
{
	if (f(arr[size - 1] > 0))
		return 1;
	return 0;
}

int calculateWithOMP(int* myArr, int mySize)
{
	int counter=0;
	int numOfThreads = omp_get_max_threads();
	int* myCounter; //array of counters for each thread
	int tid;
	int i;

	myCounter = (int*)calloc(numOfThreads, sizeof(int));

	#pragma omp parallel private(tid) //from this point, parallel the code.
	{
		tid = omp_get_thread_num();
		#pragma omp for schedule(guided, 4) //The following for loop is paralleled.
			for (i = 0; i < mySize; i++)
			{
				if (f(myArr[i]) > 0)
					myCounter[tid]++; //each thread updates his own counter.
			}
	}

	//not parallel:
	for ( i = 0; i < numOfThreads; i++)
	{
		counter += myCounter[i]; //sums all counters to one variable.
	}
	return counter;
}

int calculateWithCUDA(int* myArr, int mySize)
{
	int counter=0;
	int i;
	int *results = (int*)calloc(mySize, sizeof(int)); //results array counts positive results of f in the original array.
	if (!results)
		MPI_Abort(MPI_COMM_WORLD, 1);

	cudaError_t cudaStatus = calculateWithGPU(myArr, results, mySize); //Calls function in kernel.cu
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "calculateWithCuda failed!");
		exit(1);
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		exit(1);
	}

	for (i = 0; i < mySize; i++)
	{
		counter += results[i]; //sums results array to one variable.
	}
	free(results);

	return counter;
}

double f(int i) {
	int j;
	double value;
	double result = 0;

	for (j = 1; j < MASSIVE; j++) {
		value = (i + 1)*(j % 10);
		result += cos(value);
	}
	return cos(result);
}
