# Parallel-Programming---MPI-OMP-CUDA

A program that reads and arrays of integers from file, and calculates how many elements of said array generate a positive number when inserted into the given function f(int arr[i]);

The program seperates the array into two, allowing each process to work on half the array at the same time.
In each process the array is split to two again. 
* the first half is calculated using OMP pragmas.
* the second half is transferred to the GPU where it's calculated in parallel using CUDA functions.
