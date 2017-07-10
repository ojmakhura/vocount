/*
 * distance.cpp
 *
 *  Created on: 18 May 2016
 *      Author: junior
 */

#include "distance.hpp"
#include <omp.h>

using namespace std;

namespace clustering {
namespace distance {

template <class T>
DistanceCalculator<T>::~DistanceCalculator(){

	if(distance != NULL){
		delete [] distance;
	}

	if (coreDistances != NULL) {
		delete [] coreDistances;
	}
}

template <class T>
DistanceCalculator<T>::DistanceCalculator(calculator cal){
	this->cal = cal;
	distance = NULL;
	coreDistances = NULL;
	internalRows = internalCols = rows = cols = 0;

}

template <class T>
void DistanceCalculator<T>::setCalculator(calculator cal){
	this->cal = cal;
}

template <class T>
void DistanceCalculator<T>::computeDistance(vector<T>& dataset, int rows, int cols, bool rowwise, int minPoints) {
    setDimenstions(rows, cols);

	if (cal == _EUCLIDEAN) {

        do_euclidean(&dataset, minPoints, rowwise, true);
	} else if (cal == COSINE) {
		//do_cosine(dataset, minPoints);
	} else if (cal == _MANHATTAN) {
		//do_manhattan(dataset, minPoints);
	} else if (cal == PEARSON) {
		//do_pearson(dataset, minPoints);
	} else if (cal == SUPREMUM) {
		//do_supremum(dataset, minPoints);
	}

}

template <class T>
void DistanceCalculator<T>::computeDistance(T* dataset, int rows, int cols, bool rowwise, int minPoints){
    setDimenstions(rows, cols);

    if (cal == _EUCLIDEAN) {

        do_euclidean(&dataset, minPoints, rowwise, false);
    } else if (cal == COSINE) {
        //do_cosine(dataset, minPoints);
    } else if (cal == _MANHATTAN) {
        //do_manhattan(dataset, minPoints);
    } else if (cal == PEARSON) {
        //do_pearson(dataset, minPoints);
    } else if (cal == SUPREMUM) {
        //do_supremum(dataset, minPoints);
    }
}

template <class T>
void DistanceCalculator<T>::setDimenstions(int rows, int cols){

    this->rows = rows;
    this->cols = cols;
    int sub = (rows * rows -rows)/2;
    this->distance = new float[sub];
    this->coreDistances = new float[rows];
}

template <class T>
double DistanceCalculator<T>::getElement(void* dataset, int index, bool isVector){

    T d;
    if(isVector){
        vector<T>* v = (vector<T>*)dataset;
        d = (*v)[index];
    } else {
        T** v = (T**)dataset;
        d = (*v)[index];
    }

    return d;
}

template <class T>
void DistanceCalculator<T>::do_euclidean(void* dataset, int minPoints, bool rowwise, bool isVector){
#ifdef USE_OPENMP
#pragma omp parallel
#endif
    {

    double sortedDistance[rows];

#ifdef USE_OPENMP
#pragma omp for private(sortedDistance)
#endif
    for (uint i = 0; i < rows; i++) {
        for (uint j = 0; j < rows; j++) {
            double sum, diff = 0.0;
            uint offset2, offset1;
			sum = 0;
			//int r2, c2;
            if(rowwise){
                for (uint k = 0; ((k < cols) && (i != j)); k++) {
                    double num1 = getElement(dataset, i * cols + k, isVector);
                    double num2 = getElement(dataset, j * cols + k, isVector);
                    diff = num1 - num2;

                    sum += (diff * diff);
                }
            } else {

                double num1 = getElement(dataset, i, isVector);
                double num2 = getElement(dataset, j, isVector);
                diff = num1 - num2;

                sum += (diff * diff);
            }

			sum = sqrt(sum);

			int c;
			if (j > i) {
				// Calculate the linearised upper triangular matrix offset
				offset1 = i * rows + j;
				c = offset1 - triangular(i + 1);
				offset2 = j * rows + i;

				*(distance + c) = sum;
				//*(distance + offset2) = sum;
			} else if (i == j) {
				c = -1;
			} else {
				offset1 = j * rows + i;
				c = offset1 - triangular(j + 1);
			}

            sortedDistance[j] = sum;
			//(sortedDistance)[offset2] = sum;

		}
//#ifdef USE_OPENMP
//#pragma omp barrier
//#endif
		//#pragma omp barrier
		//roffset = i * rows;
		std::sort(sortedDistance, sortedDistance + rows);
        coreDistances[i] = sortedDistance[minPoints];
	}
    }
}

template <class T>
void DistanceCalculator<T>::computeDistance(T* dataset, int rows, int cols, int minPoints) {

	this->rows = rows;
	this->cols = cols;
	int sub = (rows * rows -rows)/2;
	this->distance = new float[sub];
	this->coreDistances = new float[rows];

	if (cal == _EUCLIDEAN) {
		do_euclidean(dataset, minPoints);
	} else if (cal == COSINE) {
		do_cosine(dataset, minPoints);
	} else if (cal == _MANHATTAN) {
		do_manhattan(dataset, minPoints);
	} else if (cal == PEARSON) {
		do_pearson(dataset, minPoints);
	} else if (cal == SUPREMUM) {
		do_supremum(dataset, minPoints);
	}

}

template <class T>
float* DistanceCalculator<T>::getDistance(){ return distance; }

template <class T>
float* DistanceCalculator<T>::getCoreDistances(){ return coreDistances; }

/**************************************************************************
 * Private methods
 **************************************************************************/

void addDistance(int i, int j, float distance){

}

template <class T>
void DistanceCalculator<T>::do_euclidean(T* dataset, int numNeighbors) {

	uint roffset, offset2, offset1;
	float sum;
	float sortedDistance[rows];

	for (uint i = 0; i < rows; i++) {
		for (uint j = 0/*i + 1*/; j < rows; j++) {
			sum = 0;

			for (uint k = 0; ((k < cols) && (i != j)); k++) {

				float num1 = *(dataset + i * cols + k);
				float num2 = *(dataset + j * cols + k);
				float diff = num1 - num2;

				sum += (diff * diff);
			}

			sum = sqrt(sum);

			int c;
			if(j > i){
				// Calculate the linearised upper triangular matrix offset
				offset1 = i * rows + j;
				c = offset1 - triangular(i + 1);
				offset2 = j * rows + i;

				*(distance + c) = sum;
			} else if(i == j){
				c = -1;
			}

			else{
				offset1 = j * rows + i;
				c = offset1 - triangular(j + 1);
			}

			(sortedDistance)[j] = sum;
		}

		std::sort(sortedDistance, sortedDistance + rows);
		coreDistances[i] = (sortedDistance)[numNeighbors];

	}
}

template <class T>
void DistanceCalculator<T>::do_cosine(T* dataset, int minPoints) {
	/*Mat dotProduct;
	Mat magnitudeOne;
	Mat magnitudeTwo;

	dotProduct = attributesOne.dot(attributesTwo);*/

	/*for (unsigned int i = 0; i < attributesOne->size() && i < attributesTwo->size(); i++) {
		dotProduct += ((*attributesOne)[i] * (*attributesTwo)[i]);
		magnitudeOne += ((*attributesOne)[i] * (*attributesOne)[i]);
		magnitudeTwo += ((*attributesTwo)[i] * (*attributesTwo)[i]);
	}*/

	//distance = 1 - (dotProduct / (magnitudeOne * magnitudeTwo));
	//reduce(distance, distance, 1, REDUCE_SUM);

}

template <class T>
void DistanceCalculator<T>::do_manhattan(T* dataset, int minPoints) {

	//absdiff(attributesOne, attributesTwo, distance);
	//reduce(distance, distance, 1, REDUCE_SUM);

	/*for (unsigned int i = 0; i < attributesOne->size() && i < attributesTwo->size(); i++) {
		distance += abs((*attributesOne)[i] - (*attributesTwo)[i]);
	}*/
}

template <class T>
void DistanceCalculator<T>::do_pearson(T* dataset, int minPoints) {

/*

	for (unsigned int i = 0; i < attributesOne.size() && i < attributesTwo->size(); i++) {
		meanOne += (*attributesOne)[i];
		meanTwo += (*attributesTwo)[i];
	}
*/

	/*reduce(attributesOne, meanOne, 1, REDUCE_SUM);
	reduce(attributesTwo, meanTwo, 1, REDUCE_SUM);

	meanOne = meanOne / attributesOne.cols;
	meanTwo = meanTwo / attributesTwo.cols;

	UMat covariance;
	UMat standardDeviationOne;
	UMat standardDeviationTwo;*/



	/*for (unsigned int i = 0; i < attributesOne->size() && i < attributesTwo->size(); i++) {
		covariance += (((*attributesOne)[i] - meanOne)
				* ((*attributesTwo)[i] - meanTwo));
		standardDeviationOne += (((*attributesOne)[i] - meanOne)
				* ((*attributesOne)[i] - meanOne));
		standardDeviationTwo += (((*attributesTwo)[i] - meanTwo)
				* ((*attributesTwo)[i] - meanTwo));
	}*/

	//distance = (1
			//- (covariance / (standardDeviationOne * standardDeviationTwo)));
}

template <class T>
void DistanceCalculator<T>::do_supremum(T* dataset, int minPoints) {

	/*for (unsigned int i = 0; i < attributesOne->size() && i < attributesTwo->size(); i++) {
		double difference = abs((*attributesOne)[i] - (*attributesTwo)[i]);
		if (difference > distance)
			distance = difference;
	}*/

}

/**
 *
 */
template <class T>
uint DistanceCalculator<T>::triangular(uint n){
	return (n * n + n)/2;
}

/**
 *
 */
template <class T>
float DistanceCalculator<T>::getDistance(uint row, uint col){

	uint idx;
	if (row < col) {
		idx = (rows * row + col) - triangular(row + 1);

	} else if (row == col) {
		return 0;
	} else {
		idx = (rows * col + row) - triangular(col + 1);
	}

	return distance[idx];
}

template class DistanceCalculator<float>;
template class DistanceCalculator<double>;
template class DistanceCalculator<int>;
template class DistanceCalculator<long>;
template class DistanceCalculator<unsigned int>;
template class DistanceCalculator<unsigned long>;
}
}
