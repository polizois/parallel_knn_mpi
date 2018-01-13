/*
    Author: Polizois Siois 8535
*/
/*
    Faculty of Electrical and Computer Engineering AUTH
    2nd assignment at Parallel and Distributed Systems (7th semester)
*/
/*
		Parallel implementation of knn algorithm using mpi.
    Data is devided in equal parts (or almost equal depending on the size
		of the data and the number of processes) and distributed to each process.
    Each process is connected to the others with ring topology and makes the
		necessary computations while sending data to the next and receiving data
		from the previous process.So there is no need for a process to have access
		to the whole set of data at once.
*/
/*
    This iteration of knn uses the NON-BLOCKING communication commands of MPI :
		- MPI_Isend
		- MPI_Irecv
		So each process can exchange data with its neighbour-processes while executing
		the knn search at the same time.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "mpi.h"

// Contains the data of a neighbour
typedef struct
{
	double idX;     // Id of the neighbour
	double dist;    // Distance between the neighbour and the reference point
	double lnn;     // Label of the neighbour
}data;

struct timeval startwtime, endwtime;
double seq_time;

double** alloc_matrix_seq_double(int rows, int cols);
data** alloc_matrix_seq_data(int rows, int cols);
double distance(double *pointA,double  *pointB,int size);
void updateDist(data *myData, int nbrs, data nbrData);
int mostFrequent(data *myData,int nbrs);
int getStart(int rank, int pointNum, int numtasks);
void copyData(double **pointsFrom, double *labelsFrom,int *rankFrom, double **pointsTo, double *labelsTo, int *rankTo, int rows, int cols);
void knnSearch(double **myPoints, int myPointNum, double **otherPoints, double *otherLabels, int otherStart, int otherPointNum, data **myData, int nbrNum, int coordNum);
int errors(data **myData, int start, int myPointNum, int pointNum, int nbrNum);
void fillDistances(data **myData, int myPointNum, int nbrNum);

int main(int argc, char *argv[])
{
		if(argc != 6){printf("Wrong arguements!!!"); return 1;}
		int pointNum = atoi(argv[1]),
		coordNum = atoi(argv[2]),
		nbrNum = atoi(argv[3]);
		char *pointsName = argv[4],
		*labelsName = argv[5];

		int i,j,t,match=0;
		FILE *pointFile;
		FILE *labelFile;
		data **myData;	// of size (taskPointNum)x(nbrNum)
		data nbrData;

		double **points, **pointsRecv, **pointsSend;  // of size (pointNum)x(coordNum) : Coordinates of every point
		double *labels, *labelsRecv, *labelsSend;   // of size (pointNum) : Labels of every point

		double singleMatch, wholeMatch;
		int singleErrorNum, wholeErrorNum;

		//mpi declarations
		int numtasks, rank, source=0, dest, tag=1, prev, next, tag1=1, tag2=2, tag3=3, rankSend, rankRecv, otherStart;

		MPI_Request reqs[6];   // required variable for non-blocking calls
    MPI_Status stats[6];   // required variable for Waitall routine



		//mpi excercising
		MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

		// If pointNum / numtasks has remainder, it means that not all processes can have the same number of points to processes
		// I choose to assign each one of the remaining points to a seperate process
		// So (pointNum % numtasks) processes will have to do the knn search for (pointNum / numtasks)+1 points,
		// while (numtasks - (pointNum % numtasks)) will have to do it for (pointNum / numtasks) points.
		int taskPointNum = pointNum / numtasks;
		int remainPointNum = pointNum % numtasks;
		int myPointNum = taskPointNum, otherPointNum = taskPointNum;
		if (remainPointNum > 0) taskPointNum+=1;

		if(rank < remainPointNum) myPointNum = taskPointNum;
		int start = getStart(rank, pointNum, numtasks);
		int end = start + myPointNum;

		// determine left and right neighbors
    prev = rank-1;
    next = rank+1;
    if (rank == 0)  prev = numtasks - 1;
    if (rank == (numtasks - 1))  next = 0;

		//Memory allocation

		//Data
		myData = alloc_matrix_seq_data(taskPointNum, nbrNum);
		fillDistances(myData, myPointNum, nbrNum);

		//Points for this task
		points = alloc_matrix_seq_double(taskPointNum, coordNum);

		//labels for this task
		labels = (double *) malloc(taskPointNum * sizeof(double));

		//Received Points
		pointsRecv = alloc_matrix_seq_double(taskPointNum, coordNum);

		//Received labels
		labelsRecv = (double *) malloc(taskPointNum * sizeof(double));

		//Sent Points
		pointsSend = alloc_matrix_seq_double(taskPointNum, coordNum);

		//Sent labels
		labelsSend = (double *) malloc(taskPointNum * sizeof(double));

		//Opening the points binary file for reading
		pointFile=fopen(pointsName,"rb");
		if (!pointFile){ printf("Unable to open file!"); return 1; }
		//Opening the labels binary file for reading
		labelFile=fopen(labelsName,"rb");
		if (!labelFile){ printf("Unable to open file!"); return 1; }

		// Loading points and labels from the .bin files to the apropriate arrays we allocated memory for

		// Finding the correct place to start loadng
		fseek(pointFile, sizeof(double)*coordNum*start, SEEK_SET);
		fseek(labelFile, sizeof(double)*1*start, SEEK_SET);

		for (i=0; i < myPointNum; i++)
		{
			//Loading a row of coordinates
			if (!fread(&points[i][0],sizeof(double),coordNum,pointFile))
			{
				printf("Unable to read from file!");
				return 1;
			}
			//Loading a label
			if(!fread(&labels[i],sizeof(double),1,labelFile))
			{
				printf("Unable to read from file!");
				return 1;
			}
		}

		//Closing the binary files
		fclose(pointFile);
		fclose(labelFile);

		//Knn execution

		// Wait all procesess to reach this point, so we can start timing
		MPI_Barrier(MPI_COMM_WORLD);
		//Timer start
		if (rank == 0) gettimeofday( &startwtime, NULL );

    //Preparing the points, labels and rank to be sent to the next process
		copyData(points, labels, &rank, pointsSend, labelsSend, &rankSend, myPointNum, coordNum);

    MPI_Irecv(&(pointsRecv[0][0]), coordNum*taskPointNum, MPI_DOUBLE, prev, tag1, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(&(labelsRecv[0]), taskPointNum, MPI_DOUBLE, prev, tag2, MPI_COMM_WORLD, &reqs[1]);
    MPI_Irecv(&rankRecv, 1, MPI_INT, prev, tag3, MPI_COMM_WORLD, &reqs[2]);

    MPI_Isend(&(pointsSend[0][0]), coordNum*taskPointNum, MPI_DOUBLE, next, tag1, MPI_COMM_WORLD, &reqs[3]);
    MPI_Isend(&(labelsSend[0]), taskPointNum, MPI_DOUBLE, next, tag2, MPI_COMM_WORLD, &reqs[4]);
    MPI_Isend(&rankSend, 1, MPI_INT, next, tag3, MPI_COMM_WORLD, &reqs[5]);

		//Running knn for the loaded points(from the bin files) with themselves
		knnSearch(points, myPointNum, points, labels, 0, myPointNum, myData, nbrNum, coordNum);

		// The knn ring data exchange
		for(t=1;t<numtasks;t++) // t=1
		{
			// wait for all non-blocking operations to complete
			MPI_Waitall(6, reqs, stats);

			// Deciding the number and the starting id of the points received
			if(rankRecv < remainPointNum) otherPointNum = taskPointNum;
			otherStart = getStart(rankRecv, pointNum, numtasks);

			// Preparin the received points, labels and rank, to be sent to the next process
			copyData(pointsRecv, labelsRecv, &rankRecv, pointsSend, labelsSend, &rankSend, otherPointNum, coordNum);

			if(t<numtasks-1) // On the last repetition there is no need to send or receive anything
			{
				MPI_Irecv(&(pointsRecv[0][0]), coordNum*taskPointNum, MPI_DOUBLE, prev, tag1, MPI_COMM_WORLD, &reqs[0]);
				MPI_Irecv(&(labelsRecv[0]), taskPointNum, MPI_DOUBLE, prev, tag2, MPI_COMM_WORLD, &reqs[1]);
				MPI_Irecv(&rankRecv, 1, MPI_INT, prev, tag3, MPI_COMM_WORLD, &reqs[2]);

				MPI_Isend(&(pointsSend[0][0]), coordNum*taskPointNum, MPI_DOUBLE, next, tag1, MPI_COMM_WORLD, &reqs[3]);
				MPI_Isend(&(labelsSend[0]), taskPointNum, MPI_DOUBLE, next, tag2, MPI_COMM_WORLD, &reqs[4]);
				MPI_Isend(&rankSend, 1, MPI_INT, next, tag3, MPI_COMM_WORLD, &reqs[5]);
			}

			// Running knn for the points received
			knnSearch(points, myPointNum, pointsSend, labelsSend, otherStart, otherPointNum, myData, nbrNum, coordNum);

		}

		// Wait all processes to finsh knn so we can get the time
		MPI_Barrier(MPI_COMM_WORLD);
		// Timer Stop
		// Gettind the time it took for the knn to complete (the time of the slowest proccess)
		if (rank == 0)
		{
			gettimeofday( &endwtime, NULL );
			seq_time = (double)( ( endwtime.tv_usec - startwtime.tv_usec ) / 1.0e6 + endwtime.tv_sec - startwtime.tv_sec );
		}

		//Error check with results from matlab (count process errors)
		singleErrorNum = errors(myData, start, myPointNum, pointNum, nbrNum);

		//find most frequent neighbour label of every point
		//and count the matches with the original labels
		for(i=0;i<myPointNum;i++)
			if((double) mostFrequent(myData[i], nbrNum) == labels[i]) match++;

		// Calculate match percentage for this process
		singleMatch = ((double)match) / myPointNum * 100;

		MPI_Barrier(MPI_COMM_WORLD);
		// Gather the match percentages of ever process and sumate them (in process 0)
		MPI_Reduce (&singleMatch,&wholeMatch,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
		// Gather errors made by every process and samate them (in process 0)
		MPI_Reduce (&singleErrorNum,&wholeErrorNum,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);


		// Displaying results
		if(rank == 0)
		{
			printf("#### Knn - non-blocking version ####\n");
			printf("Points : %d / Coordinates : %d / Nearest neighbours : %d\n", pointNum, coordNum, nbrNum);
			printf("Whole knn done in %.2f sec\n", seq_time);
			printf("!!!! Whole match percentage : %3.1f%% !!!!\n", wholeMatch/numtasks);
			printf("!!!! Whole errors (based on matlab) : %d !!!!\n", wholeErrorNum);
		}

		MPI_Finalize();

		return 0;
}

// Calculates and return the euclidean distance between 2 points
double distance(double *pointA, double  *pointB, int size)
{
	int i;
	double sum=0;
	for(i=0;i<size;i++)
	{
		sum += pow(pointA[i]-pointB[i], 2);
	}

	return sqrt(sum);
}

// Places(if needed) a point(nbrData) in the correct place of another point's nearest-neighbour row in myData
void updateDist(data *myData, int nbrs, data nbrData)
{
	int i,pos = -1;
	data tempData;
	data tempNext, tempCurr;

	for(int i=0; i<nbrs; i++)
	{
		if(myData[i].dist == -1)
			{ myData[i] = nbrData; return; }

		else if(nbrData.dist <= myData[i].dist)
			{ pos = i; break; }
	}
	if (pos != -1)
	{
		tempNext = nbrData;
		for(int i=pos; i<nbrs; i++)
		{
			tempCurr = tempNext;
			tempNext = myData[i];
			myData[i] = tempCurr;
		}

		return;
	}

	return;
}

// Finds the most frequent label of the nearest-neighbours of a specific point
int mostFrequent(data *myData,int nbrs)
{
	int i,j,count=0,tempCount,label,tempLabel;
	int checked[nbrs],checkedCount=0, skip=0;

	for(i=0;i<nbrs;i++)
	{
		tempLabel = (int)myData[i].lnn;
		tempCount = 0;

		skip = 0;
		for(j=0;j<checkedCount;j++)
		{
			if(tempLabel == checked[j])
			{
				skip = 1;
				break;
			}
		}

		if(!skip)
		{
			for(j=0;j<nbrs;j++)
			{
				if((int)myData[j].lnn == tempLabel)
					tempCount++;
			}

			if(tempCount >= count)
			{
				count = tempCount;
				label = tempLabel;
			}

			checked[checkedCount++] = label;
		}
	}

	return label;
}

// Allocates continuous memory for a 2d array of doubles
double** alloc_matrix_seq_double(int rows, int cols)
{
	int i;

	double **matrix= malloc(rows * sizeof(*matrix));
	if(!matrix)
	{
   printf("Out of memory\n");
   exit(-1);
 }
	matrix[0] = malloc(rows * (cols) * sizeof(**matrix));
	if(!matrix[0])
	{
   printf("Out of memory\n");
   exit(-1);
 }
	for(i = 1; i < rows; i++)
		matrix[i] = matrix[0] + i * (cols);

	return matrix;
}

// Allocates continuous memory for a 2d array of type data
data** alloc_matrix_seq_data(int rows, int cols)
{
	int i;

	data **matrix= malloc(rows * sizeof(*matrix));
	if(!matrix)
	{
   printf("Out of memory\n");
   exit(-1);
 }
	matrix[0] = malloc(rows * (cols) * sizeof(**matrix));
	if(!matrix[0])
	{
   printf("Out of memory\n");
   exit(-1);
 }
	for(i = 1; i < rows; i++)
		matrix[i] = matrix[0] + i * (cols);

	return matrix;
}

// Returns the correct id of the first point of a process
int getStart(int rank, int pointNum, int numtasks)
{
	int remain = pointNum % numtasks;
	int taskPoints = pointNum / numtasks;

	if(rank < remain) return rank*taskPoints+rank;
	else return (remain*taskPoints+remain) + (rank-remain)*taskPoints;
}

// Copies a points table, a labels array and a rank interger
void copyData(double **pointsFrom, double *labelsFrom,int *rankFrom, double **pointsTo, double *labelsTo, int *rankTo, int rows, int cols)
{
	int i,j;
	for(i=0;i<rows;i++)
	{
		for(j=0;j<cols;j++)
			pointsTo[i][j] = pointsFrom[i][j];
		labelsTo[i] = labelsFrom[i];
	}
	*rankTo = *rankFrom;
}

// Does the knn knn search
// Uses as reference the myPoints table
// Uses otherPoints table for potetial nearest neighbours of the myPoint points
// Updates myData with the changes needed
void knnSearch(double **myPoints, int myPointNum, double **otherPoints, double *otherLabels, int otherStart, int otherPointNum, data **myData, int nbrNum, int coordNum)
{
	int i,j;
	data nbrData;

	if(myPoints == otherPoints)
	{
		for(i=0; i<myPointNum-1; i++)
		{
			for(j=i+1; j<otherPointNum; j++)
			{
					nbrData.dist = distance(myPoints[i], otherPoints[j], coordNum);
					nbrData.lnn = otherLabels[j];
					nbrData.idX = otherStart+j;
					updateDist(myData[i], nbrNum, nbrData);

					nbrData.lnn = otherLabels[i];
					nbrData.idX = otherStart+i;
					updateDist(myData[j], nbrNum, nbrData);
			}
		}
	}
	else
	{
		for(i=0; i<myPointNum; i++)
		{
			for(j=0; j<otherPointNum; j++)
			{
					nbrData.dist = distance(myPoints[i], otherPoints[j], coordNum);
					nbrData.lnn = otherLabels[j];
					nbrData.idX = otherStart+j; //start+j
					updateDist(myData[i], nbrNum, nbrData);
			}
		}
	}

}

// Opens a binary file(produced by knn search on matlab) that has the labels of ne k nearest neighbours of every point
// Checks the labels found in myData for errors, counnts the errors and returns them
int errors(data **myData, int start, int myPointNum, int pointNum, int nbrNum)
{
	FILE *labelResults;
	double *tempLine;
	char fileName[100];
	int i,j, er=0;

	//Generating the file name
	sprintf(fileName, "./confirm/lnn_%d_%d.bin", pointNum, nbrNum);
	// Allocating space for the reading line
	tempLine = (double *) malloc(nbrNum * sizeof(double));

	//Opening the label results binary file for reading
	labelResults=fopen(fileName,"rb");
	if (!labelResults){ printf("Unable to open file!"); return 1; }

	// Finding the correct place to start loadng
	fseek(labelResults, sizeof(double)*nbrNum*start, SEEK_SET);

	// reading every line and checking if theres a difference between my results and those from matlab
	for (i=0; i < myPointNum; i++)
	{
		//Loading a label
		if(!fread(tempLine, sizeof(double), nbrNum, labelResults))
			{ printf("Unable to read from file!"); return 1; }
		for(j=0;j<nbrNum;j++)
			if(tempLine[j] != myData[i][j].lnn) { er++; break; }
	}
	//Closing the binary files
	fclose(labelResults);

	return er;
}

// Fills the distances of the nearest neighbours of every point with -1, which signifies an empty place
void fillDistances(data **myData, int myPointNum, int nbrNum)
{
	int i, j;
	for(i=0;i<myPointNum;i++)
	{
		for(j=0;j<nbrNum;j++)
			myData[i][j].dist = -1;
	}
}
