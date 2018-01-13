# Parallel knn using Mpi
Assignement for the course named "Parrallel and Distributed Systems" of Aristotle University's "Electrical and Computer Engineering" department.

to compile the blocking version run the following command :  
***mpicc ./knn_mpi_block.c -o knn_mpi_block.out -O3 -lm***  
to compile the non-blocking version run the following command :  
***mpicc ./knn_mpi_non.c -o knn_mpi_non.out -O3 -lm***  

The program takes 5 arguements to run correctly:  
  * arg1 : Number of points  
  * arg2 : Number of coordinates of a pint  
  * arg3 : Number of nearest neighbours  
  * arg4 : Path for the points binary file  
  * arg5 : Path for the labels binary file  

The data folder contains the data on which the programs run.  
There are 2 data sets :   
  * 1 : file with 60000 points of 30 coordinates each + file with the labels of every point (points_60000_30.bin + labels_60000_30.bin)  
  * 2 : file with 10000 points of 784 coordinates each + file with the labels of every point (points_10000_784.bin + labels_10000_784.bin)  

The confirm folder contains results from knn run on matlab on both of the data sets, for 5 and 10 nearest neighbours
These bin files are used by the program in order to confirm that its results are correct.

Let's assume you want to run the program on the first data set, for 5 nearest neighbours and for 4 processes.
In order to run the program localy, if the files are positioned exactly like they are on the repo, you can run the following commands (after, of course, you have compiled) :  

blocking version:  
***mpiexec -np 4 ./knn_mpi_block.out 60000 30 5 ./data/points_60000_30.bin ./data/labels_60000_30.bin***  
non-blocking version:  
***mpiexec -np 4 ./knn_mpi_non.out 60000 30 5 ./data/points_60000_30.bin ./data/labels_60000_30.bin***  
	
After the program has finished running, you sould be presentet with :  
  * the time it took for the knn to run  
  * a match percentage (an evaluation of how well the algorithm works)  
  * an error number which should be 0 (if not 0, then there is some problem with the program and it doesn't give the same results as matlab)  
