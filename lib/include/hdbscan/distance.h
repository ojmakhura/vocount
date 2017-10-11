/*
 * distance.h
 *
 *  Created on: 18 May 2016
 *      Author: junior
 */

#ifndef DISTANCE_H_
#define DISTANCE_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <omp.h>
#include <math.h>
#include <stdint.h>
typedef unsigned int uint;

#define DISTANCE_SUCCESS 1
#define DISTANCE_ERROR 0

#ifdef __cplusplus
namespace clustering {
#endif

#ifndef COSINE
#define COSINE 			0
#endif

#ifndef _EUCLIDEAN
#define _EUCLIDEAN 		1
#endif

#define DATATYPE_FLOAT			0
#define DATATYPE_DOUBLE			1
#define DATATYPE_INT			2
#define DATATYPE_LONG			3
#define DATATYPE_SHORT			4

/*
#ifndef MANHATTAN
#define MANHATTAN		2
#endif
*/
/*

#ifndef PEARSON
#define PEARSON			3
#endif

#ifndef SUPREMUM
#define SUPREMUM		4
#endif
*/

typedef unsigned int calculator;

struct Distance{
	double* distances;
	double* coreDistances;
	uint rows, cols;
	uint internalRows, internalCols;
	int numNeighbors;
	calculator cal;
	uint datatype;

#ifdef __cplusplus
public:
	Distance(calculator cal);
	Distance();
	~Distance();

	/**
	 * Retrieves the distance from the array. Because the array
	 * is an upper triangular matrix stored as a 1 dimensional array
	 *
	 */
    double getDistance(uint row, uint col);

private:

	/**
	 * Calculates the n_th triangular number
	 *
	 * @param n
	 *
	 */

	uint triangular(uint n);
    void setDimenstions(uint rows, uint cols);
	/**
	 * Computes the euclidean distance between two points, d = sqrt((x1-y1)^2 + (x2-y2)^2 + ... + (xn-yn)^2).
	 */
	void doEuclidean(void* dataset, int minPoints);


#endif
};

/*
#ifdef __cplusplus
template <class T>
class Distance Distance;
#endif
*/

//typedef struct Distance Distance;
typedef struct Distance distance;

distance* distance_init(distance* dis, calculator cal, uint datatype);
void distance_destroy(distance* d);
void distance_clean(distance* d);

/**
 * Computes the euclidean distance between two points, d = sqrt((x1-y1)^2 + (x2-y2)^2 + ... + (xn-yn)^2).
 */
void distance_euclidean(void* dataset, distance* dis, uint rows, uint cols);

double distance_get(distance* dis, uint row, uint col);
void distance_compute(distance* dis, void* dataset, int rows, int cols, int numNeighbors);

#ifdef __cplusplus
};
}
#endif
#endif /* DISTANCE_H_ */
