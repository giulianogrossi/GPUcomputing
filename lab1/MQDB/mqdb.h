/*
 * mqdb.h
 *
 * Matrices are stored in row-major order:
 * M(i,j) corresponds to *(M.elem + i * M.cols + j)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#ifndef MQDB_H
#define MQDB_H

#define randu() ((float)rand() / (float) RAND_MAX)
#define abs(x) ((x)<0 ? (-x) : (x))

typedef unsigned long ulong;
typedef unsigned int uint;

typedef struct MQDB {
	char desc[100];   // description
	int nBlocks;      // num. of blocks
	int *blkSize;     // block dimensions
	float *elem;      // elements in row-major order
	ulong nElems;     // actual number of elements
} mqdb;


inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

typedef unsigned long ulong;
typedef unsigned int uint;

// function prototypes
int genRandDims(mqdb*, uint, uint);
void fillBlocks(mqdb*, uint, uint, char, float);
mqdb mqdbConst(uint, uint, uint, float);
void mqdbProd(mqdb, mqdb, mqdb);
void matProd(mqdb, mqdb, mqdb);
void checkResult(mqdb, mqdb);
void mqdbDisplay(mqdb);

#endif
