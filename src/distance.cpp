/*
 * distance.cpp
 *
 *  Created on: 18 May 2016
 *      Author: junior
 */

#include "distance.hpp"
#include <tbb/tbb.h>
#include <omp.h>

using namespace std;

namespace clustering {
namespace distance {

DistanceCalculator::~DistanceCalculator(){

	if(distance != NULL){
		delete [] distance;
	}

	if (coreDistances != NULL) {
		delete [] coreDistances;
	}
}

DistanceCalculator::DistanceCalculator(calculator cal){
	this->cal = cal;
	distance = NULL;
	coreDistances = NULL;
	internalRows = internalCols = rows = cols = 0;

}

void DistanceCalculator::setCalculator(calculator cal){
	this->cal = cal;
}

void DistanceCalculator::cvComputeEuclidean(vector<vector<Point> > contours, Mat flow, Mat frame, int numNeighbors){

	CV_Assert(flow.isContinuous());
	CV_Assert(frame.isContinuous());
	CV_Assert(frame.cols == flow.cols);
	CV_Assert(frame.rows == flow.rows);

	rows = 0;
	cols = 4 + frame.channels();

	/**
	 * Calculate the total number of contour points
	 */
	for (size_t i = 0; i < contours.size(); ++i) {
		rows += contours[i].size();
	}

	printf("found %d points\n", rows);
	this->cols = frame.channels() + 4;
	//this->distance = (float *) malloc(rows * rows * sizeof(float));
	this->coreDistances =  (float *) malloc(rows * sizeof(float));

	if (cal == _EUCLIDEAN) {
		cvEuclidean(contours, flow, frame, numNeighbors);
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

void DistanceCalculator::cvComputeEuclidean(vector<Point> contour, vector<Mat>& dataset, int numNeighbors, bool includeIndex){

	rows = contour.size();
	cols = 2 + dataset.size();

	printf("found %d points\n", rows);
	int sub = (rows * rows -rows)/2;
	//this->cols = frame.channels() + 4;
	//this->distance = (float *) malloc(rows * rows * sizeof(float));
	this->coreDistances = new float[rows];
	this->distance = new float[sub];

	if (cal == _EUCLIDEAN) {
		do_euclidean(contour, &dataset, numNeighbors, includeIndex);
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


void DistanceCalculator::computeDistance(vector<vector<float> > dataset, int numNeighbors){
	CV_Assert(dataset.size() > 0);
	this->rows = dataset[0].size();
	int sub = (rows * rows - rows) / 2;
	printf("found %d points and sub %d\n", rows, sub);
	// dataset only contains the data matrices
	//
	this->cols = dataset.size();

	this->distance = new float[sub];
	//this->sortedDistance = (float *) malloc(rows * rows * sizeof(float));
	this->coreDistances = new float[rows];

	do_euclidean(dataset, numNeighbors);
}

void DistanceCalculator::cvComputeDistance(vector<Mat>& dataset, int minPoints, bool indexed){
	CV_Assert(dataset.size() > 0);

	/**
	 * Need all Matrices to be continuous and have only one channel
	 */
#pragma omp parallel for
	for(uint x = 0; x < dataset.size(); ++x){
		CV_Assert(dataset[x].isContinuous());
		//CV_Assert(dataset[x].channels() == 1);
	}

	this->rows = dataset[0].rows * dataset[0].cols;
	int sub = (rows * rows -rows)/2;
	printf("found %d points and sub %d\n", rows, sub);
	// dataset only contains the data matrices
	//
	this->cols = dataset.size() + 2;

	this->distance = new float[sub];
	//this->sortedDistance = (float *) malloc(rows * rows * sizeof(float));
	this->coreDistances = new float[rows];

	if (cal == _EUCLIDEAN) {
		//cvEuclidean(&dataset, minPoints);
		do_euclidean(&dataset, minPoints, indexed);
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

void DistanceCalculator::cvComputeDistance(Mat& dataset, int minPoints){


	this->rows = dataset.rows;
	int sub = (rows * rows -rows)/2;
	printf("found %d points and sub %d\n", rows, sub);
	// dataset only contains the data matrices
	//
	this->cols =  dataset.cols;

	this->distance = new float[sub];
	//this->sortedDistance = (float *) malloc(rows * rows * sizeof(float));
	this->coreDistances = new float[rows];

	if (cal == _EUCLIDEAN) {
		//cvEuclidean(&dataset, minPoints);
		do_euclidean(dataset, minPoints);
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

void DistanceCalculator::computeDistance(vector<Point2f> dataset, int rows, int cols, int minPoints) {

	this->rows = rows;
	this->cols = cols;
	int sub = (rows * rows -rows)/2;
	printf("rows is %d sub is %d\n", rows, sub);
	//printf("requested memory = %d bytes\n", rows * rows * sizeof(float));
	this->distance = new float[sub]; //(float *) malloc(sub * sizeof(float));
	//this->sortedDistance = (float *) malloc(rows * rows * sizeof(float));
	this->coreDistances = new float[rows]; // (float *) malloc(rows * sizeof(float));

	//printf("(distance, sortedDistance, coreDistances) = (%d, %d, %d)", distance, sortedDistance, coreDistances);

	/*if (cal == _EUCLIDEAN) {
		do_euclidean(dataset, minPoints);
	} else if (cal == COSINE) {
		do_cosine(dataset, minPoints);
	} else if (cal == _MANHATTAN) {
		do_manhattan(dataset, minPoints);
	} else if (cal == PEARSON) {
		do_pearson(dataset, minPoints);
	} else if (cal == SUPREMUM) {
		do_supremum(dataset, minPoints);
	}*/

}

void DistanceCalculator::computeDistance(float* dataset, int rows, int cols, int minPoints) {

	this->rows = rows;
	this->cols = cols;
	int sub = (rows * rows -rows)/2;
	printf("rows is %d sub is %d\n", rows, sub);
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

float* DistanceCalculator::getDistance(){ return distance; }
//float* DistanceCalculator::getSortedDistance(){ return sortedDistance; }
float* DistanceCalculator::getCoreDistances(){ return coreDistances; }

/**************************************************************************
 * Private methods that handle opencv data
 **************************************************************************/

void DistanceCalculator::cvEuclidean(vector<Mat>* dataset, int numNeighbors){

	printf("DistanceCalculator::cvEuclidean\n");

	do_euclidean(dataset, numNeighbors, true);
}

void DistanceCalculator::do_euclidean(vector<vector<float> >& dataset, int minPoints){
	float sortedDistance[rows];
	for (uint i = 0; i < rows; i++) {
		//float* p1 = dataset.ptr<float>(i);
		for (uint j = 0; j < rows; j++) {
			//float* p2 = dataset.ptr<float>(j);
			float sum, diff = 0.0;
			uint roffset, offset2, offset1;
			sum = 0;
			//int r2, c2;

			for (uint k = 0; ((k < cols) && (i != j)); k++) {
				float num1 = dataset[k][i];
				float num2 = dataset[k][j];
				diff = num1 - num2;

				sum += (diff * diff);
				//}
			}

			sum = sqrt(sum);

			//printf("sum = %f\n", sum);
			int c;
			if (j > i) {
				// Calculate the linearised upper triangular matrix offset
				offset1 = i * rows + j;
				c = offset1 - triangular(i + 1);
				//printf("c calculated %d from (%d, %d)\n", c, i, j);
				offset2 = j * rows + i;
				//printf("offset calculated at (%d, %d)\n", offset1, offset2);

				*(distance + c) = sum;
				//printf("*(distance + c) 1 alloc %f\n", *(distance + c));
				//*(distance + offset2) = sum;
				//printf("distance sum set\n");
			} else if (i == j) {
				c = -1;
			} else {
				offset1 = j * rows + i;
				c = offset1 - triangular(j + 1);
			}

			(sortedDistance)[j] = sum;
			//(sortedDistance)[offset2] = sum;
			//printf("%.2f \n", *(distance + offset1));

			//printf("Index : %d ------ > (%d, %d) = %f\n", c, i, j, sum);
		}
		//#pragma omp barrier
		//roffset = i * rows;
		std::sort(sortedDistance, sortedDistance + rows);
		coreDistances[i] = (sortedDistance)[minPoints];
	}
}

void DistanceCalculator::do_euclidean(Mat& dataset, int numNeighbors){


	//distance = 0;
	//printf("calculating distance (%d, %d, %f)\n", rows, cols, *dataset);

#ifdef USE_OPENMP
#pragma omp parallel
#endif
	{

	float sortedDistance[rows];
	int r1 = 0, c1 = 0;
#ifdef USE_OPENMP
#pragma omp for
#endif
	for (uint i = 0; i < rows; i++) {
		float* p1 = dataset.ptr<float>(i);
		for (uint j = 0; j < rows; j++) {
			float* p2 = dataset.ptr<float>(j);
			float sum, diff = 0.0;
			uint roffset, offset2, offset1;
			sum = 0;
			int r2, c2;
			/*if(points.size() > 0){
				vector<float> v2 = points[2];
				vector<float> v1 = points[i];

				for(uint k = 0; k < v2.size(); ++k){
					float num1 = v1[k];
					float num2 = v2[k];
					diff = num1 - num2;

					//printf("inner peace %d: (%f - %f) \n", k, num1, num2);

					sum += (diff * diff);
				}
			}*/

			for (uint k = 0; ((k < cols) && (i != j)); k++) {
				float num1 = p1[k];
				float num2 = p2[k];
				diff = num1 - num2;

				//printf("inner peace %d: (%f - %f) \n", k, num1, num2);

				sum += (diff * diff);
				//}
			}

			sum = sqrt(sum);

			//printf("sum = %f\n", sum);
			int c;
			if(j > i){
				// Calculate the linearised upper triangular matrix offset
				offset1 = i * rows + j;
				c = offset1 - triangular(i + 1);
				//printf("c calculated %d from (%d, %d)\n", c, i, j);
				offset2 = j * rows + i;
				//printf("offset calculated at (%d, %d)\n", offset1, offset2);

				*(distance + c) = sum;
				//printf("*(distance + c) 1 alloc %f\n", *(distance + c));
				//*(distance + offset2) = sum;
				//printf("distance sum set\n");
			} else if(i == j){
				c = -1;
			} else{
				offset1 = j * rows + i;
				c = offset1 - triangular(j + 1);
			}

			(sortedDistance)[j] = sum;
			//(sortedDistance)[offset2] = sum;
			//printf("%.2f \n", *(distance + offset1));

			//printf("Index : %d ------ > (%d, %d) = %f\n", c, i, j, sum);
		}
//#pragma omp barrier
		//roffset = i * rows;
		std::sort(sortedDistance, sortedDistance + rows);
		coreDistances[i] = (sortedDistance)[numNeighbors];
	}
	}

}

void DistanceCalculator::cvEuclidean(vector<vector<Point> > contours, Mat flow, Mat frame, int numNeighbors){
	printf("DistanceCalculator::cvEuclidean(vector<vector<Point> > contours, Mat flow, Mat frame, int numNeighbors)\n");
	uint roffset, offset2, offset1;
	float sum;

	//Point2f* flowData = flow.

	uint conCount1 = 0, pCount1 = 0;

	uchar* data = frame.data;

	for (uint i = 0; i < rows; i++) {

		Point p1 = contours[conCount1][pCount1];
		uint conCount2 = 0, pCount2 = 0;
		float num1, num2, diff;
		for (uint j = i; j < rows; j++) {
			//tbb::parallel_for(size_t(0), size_t(rows), [=](size_t j){
			if(i != j){
				sum = 0;

				//printf("calculating for mul: (%d, %d)\n", conCount2, pCount2);

				/**
				 * Idices
				 */
				Point p2 = contours[conCount2][pCount2];

				//printf("Point 1 (%d, %d) : Point 2 (%d, %d)\n", p1.x, p1.y, p2.x, p2.y);

				num1 = p1.x;
				num2 = p2.x;
				diff = num1 - num2;
				sum += (diff * diff);

				num1 = p1.y;
				num2 = p2.y;
				diff = num1 - num2;
				sum += (diff * diff);

				/**
				 * Flow points
				 *
				 * If flow has been provided
				 */
				if (!flow.empty()) {
					Point2f f1 = flow.at<Point2f>(p1);
					Point2f f2 = flow.at<Point2f>(p2);

					num1 = f1.x;
					num2 = f2.x;
					//printf("(f1.x, f2.x) = (%2f, %2f)", num1, num2);
					diff = num1 - num2;
					sum += (diff * diff);

					num1 = f1.y;
					num2 = f2.y;
					//printf("(f1.y, f2.y) = (%2f, %2f)", num1, num2);
					diff = num1 - num2;
					sum += (diff * diff);
				}

				/**
				 * Frame pixels
				 *
				 * Depending on how many channels the frame has,
				 * we add additional processing for the distance columns
				 */
				uchar v1, v2;
				switch (frame.channels()) {
				case 1:
					v1 = frame.at<uchar>(p1);
					v2 = frame.at<uchar>(p2);

					diff = v1 - v2;
					sum += (diff * diff);

					break;
				case 3:
					/**
					 * When the frame is coloured, we access the
					 * matrix as a Vec3b matrix
					 */
					Vec3b vec1 = frame.at<Vec3b>(p1);
					Vec3b vec2 = frame.at<Vec3b>(p2);

					// Blue
					num1 = vec1[0];
					num2 = vec2[0];
					diff = num1 - num2;
					sum += (diff * diff);

					// Green
					num1 = vec1[1];
					num2 = vec2[1];
					diff = num1 - num2;
					sum += (diff * diff);

					//Red
					num1 = vec1[2];
					num2 = vec2[2];
					diff = num1 - num2;
					sum += (diff * diff);
					break;
				}
			}

			/**
			 * When calculating distances using the same matrix,
			 * opposite pixels (pixels mirrored on the diagonal)
			 * have the same values with the diagonal being 0
			 * so to reduce computation, we only compute the top
			 * half and copy to the mirror pixels.
			 */
			//printf("sum = %f\n", sum);
			offset1 = i * rows + j;
			//printf("offset1 calculated\n");
			offset2 = j * rows + i;
			//printf("offset calculated at (%d, %d)\n", offset1, offset2);

			*(distance + offset1) = sum;
			//printf("offset 1 alloc\n");
			*(distance + offset2) = sum;
			//printf("distance sum set\n");

			//*(sortedDistance + offset1) = sum;
			//*(sortedDistance + offset2) = sum;
			//printf("%.2f \n", *(distance + offset1));

			/**
			 * The contours are being accessed point by point so
			 * we need some clever way of going from point to point.
			 * When we reach the end of the contour, we just advance
			 * next contour and start at the beginning
			 */
			//printf("Contour number %d has size %d\n", conCount2,contours[conCount2].size() );
			if(pCount2 < contours[conCount2].size()-1){
				++pCount2;
			} else{
				pCount2 = 0;
				++conCount2;
			}

		}

		roffset = i * rows;
		//std::sort(sortedDistance + roffset, sortedDistance + roffset + rows);
		//coreDistances[i] = *(sortedDistance + roffset + point);
		//printf("coreDistances[%f] = %d\n", i, coreDistances[i]);

		if(pCount1 < contours[conCount1].size()-1){
			++pCount1;
		} else{
			pCount1 = 0;
			++conCount1;
		}
	}

}

/**************************************************************************
 * Private methods
 **************************************************************************/

void addDistance(int i, int j, float distance){

}

float convolution(Point p, Mat m, int kernelSize){

	float sum = 0.0;
	int red = kernelSize/2;

	for(int i = 0; i < kernelSize; i++){
		for(int j = 0; j < kernelSize; j++){
			int r = p.y + i - red;

			if(r < 0){
				r = p.y;
			}

			int c = p.x + j - red;

			if(c < 0){
				c = p.x;
			}

			if(m.depth() == CV_8U){
				sum += m.at<uchar>(r, c);
			} else if(m.depth() == CV_32F){
				sum += m.at<float>(r, c);
			}
		}
	}

	return sum/(kernelSize * kernelSize);
}

void DistanceCalculator::do_euclidean(vector<Point> contour, vector<Mat>* dataset, int numNeighbors, bool includeIndex) {
	//distance = 0;
	//printf("calculating distance (%d, %d, %f)\n", rows, cols, *dataset);
	printf("thread number %d\n", omp_get_thread_num());
	uint roffset, offset2, offset1;
	float sum, diff;
	float sortedDistance[rows];

	int r1, c1, r2, c2;

	for (uint i = 0; i < rows; i++) {
		r1 = contour[i].y;
		c1 = contour[i].x;

		for (uint j = 0; j < rows; j++) {
			sum = 0;
			r2 = contour[j].y;
			c2 = contour[j].x;

			if(includeIndex){
				//printf("(%d, %d)\t", r1, c1);
				//printf("(%d, %d)\t", r2, c2);
				float num1 = r1 - r2;// = *(dataset + i * cols + k);
				float num2 = c1 - c2;
				sum += (num1 * num1);
				sum += (num2 * num2);
				//printf("calculating for mul: (%d, %d)\n", i, j);
			}

			int colOffset = 2;

			if(includeIndex){
				colOffset = 2;
			}

			for (uint k = 0; ((k < dataset->size()) && (i != j)); k++) {

				//for(uint x = 0; x < dataset->size(); ++x){

					Mat m = (*dataset)[k];
					offset1 = r1 * m.cols + c1;
					offset2 = r2 * m.cols + c2;

					if(m.depth() == CV_8U){
						//printf("I am 8U\n");

						//printf("CV_8U at %d %d and %d %d\n", r1, c1, r2, c2);

						uchar* data = m.data;

						float num1 = *(data + offset1);
						float num2 = *(data + offset2);

						//num1 = convolution(Point(r1, c1), m, 3);
						//num1 = convolution(Point(r2, c2), m, 3);

						diff = num1 - num2;
					} else if(m.depth() == CV_32F){
						//printf("I am 32S\n");
						float* data = (float*)m.ptr<float>(0);
						float num1 = *(data + offset1);
						float num2 = *(data + offset2);
						//num1 = convolution(Point(r1, c1), m, 3);
						//num1 = convolution(Point(r2, c2), m, 3);
						diff = num1 - num2;

					}

					//printf("inner peace %d: (%f - %f) \n", k, num1, num2);

					sum += (diff * diff);
				//}
			}

			sum = sqrt(sum);

			//printf("sum = %f\n", sum);
			int c;
			if(j > i){
				// Calculate the linearised upper triangular matrix offset
				offset1 = i * rows + j;
				c = offset1 - triangular(i + 1);
				//printf("c calculated %d from (%d, %d)\n", c, i, j);
				offset2 = j * rows + i;
				//printf("offset calculated at (%d, %d)\n", offset1, offset2);

				*(distance + c) = sum;
				//printf("*(distance + c) 1 alloc %f\n", *(distance + c));
				//*(distance + offset2) = sum;
				//printf("distance sum set\n");
			} else if(i == j){
				c = -1;
			}

			else{
				offset1 = j * rows + i;
				c = offset1 - triangular(j + 1);
			}

			(sortedDistance)[j] = sum;
			//(sortedDistance)[offset2] = sum;
			//printf("%.2f \n", *(distance + offset1));

			//printf("Index : %d ------ > (%d, %d) = %f\n", c, i, j, sum);
		}

		//roffset = i * rows;
		std::sort(sortedDistance, sortedDistance + rows);
		coreDistances[i] = (sortedDistance)[numNeighbors];

		/*for(int s = 0; s < 20; ++s){
			printf("%f, ", sortedDistance[s]);
		}
		printf("\n");*/

		//printf("coreDistances[%f] = %d\n", i, coreDistances[i]);
		//delete sortedDistance;
	}

	/*printf("\n*************** Printing Distance Matrix ******************\n");

	int n = (rows*rows - rows)/2;
	printf("n is %d\n", n);
	for (int i = 0; i < n; i++) {
		printf("%d	|	%.1f\t\n", i,  *(distance + i));
	}*/
	/*
	printf(
			"\n*************** Printing Sorted Distance Matrix ******************\n");

	for (int i = 0; i < rows; i++) {
		printf("%.1f\t", *(coreDistances + i));

		printf("\n");
	}*/

	//printf("done");

}

void DistanceCalculator::do_euclidean(vector<Mat>* dataset, int numNeighbors, bool includeIndex) {
	//distance = 0;
	//printf("calculating distance (%d, %d, %f)\n", rows, cols, *dataset);

#ifdef USE_OPENMP
#pragma omp parallel
#endif
	{

	float sortedDistance[rows];
	int r1 = 0, c1 = 0;
#ifdef USE_OPENMP
#pragma omp for
#endif
	for (uint i = 0; i < rows; i++) {

		if(includeIndex){
			r1 = i/(*dataset)[0].cols;
			c1 = i%(*dataset)[0].cols;
		}

		for (uint j = 0; j < rows; j++) {
			float sum, diff = 0.0;
			uint roffset, offset2, offset1;
			sum = 0;
			int r2, c2;
			if(includeIndex){
				r2 = j/(*dataset)[0].cols;
				c2 = j%(*dataset)[0].cols;
				//printf("(%d, %d)\t", r1, c1);
				//printf("(%d, %d)\t", r2, c2);
				float num1 = r1 - r2;// = *(dataset + i * cols + k);
				float num2 = c1 - c2;
				sum += (num1 * num1);
				sum += (num2 * num2);
				//printf("calculating for mul: (%d, %d)\n", i, j);
			}

			int colOffset = 2;

			if(includeIndex){
				colOffset = 2;
			}

			for (uint k = 0; ((k < cols - colOffset) && (i != j)); k++) {

				//for(uint x = 0; x < dataset->size(); ++x){

					Mat m = (*dataset)[k];

					if(m.depth() == CV_8U){
						//printf("I am 8U\n");

						//printf("CV_8U at %d %d and %d %d\n", r1, c1, r2, c2);

						uchar* data = m.data;
						float num1 = *(data + i);
						float num2 = *(data + j);
						diff = num1 - num2;
					} else if(m.depth() == CV_32F){
						//printf("I am 32S\n");
						float* data = (float*)m.ptr<float>(0);
						float num1 = *(data + i);
						float num2 = *(data + j);
						diff = num1 - num2;

					}

					//printf("inner peace %d: (%f - %f) \n", k, num1, num2);

					sum += (diff * diff);
				//}
			}

			sum = sqrt(sum);

			//printf("sum = %f\n", sum);
			int c;
			if(j > i){
				// Calculate the linearised upper triangular matrix offset
				offset1 = i * rows + j;
				c = offset1 - triangular(i + 1);
				//printf("c calculated %d from (%d, %d)\n", c, i, j);
				offset2 = j * rows + i;
				//printf("offset calculated at (%d, %d)\n", offset1, offset2);

				*(distance + c) = sum;
				//printf("*(distance + c) 1 alloc %f\n", *(distance + c));
				//*(distance + offset2) = sum;
				//printf("distance sum set\n");
			} else if(i == j){
				c = -1;
			} else{
				offset1 = j * rows + i;
				c = offset1 - triangular(j + 1);
			}

			(sortedDistance)[j] = sum;
			//(sortedDistance)[offset2] = sum;
			//printf("%.2f \n", *(distance + offset1));

			//printf("Index : %d ------ > (%d, %d) = %f\n", c, i, j, sum);
		}
//#pragma omp barrier
		//roffset = i * rows;
		std::sort(sortedDistance, sortedDistance + rows);
		coreDistances[i] = (sortedDistance)[numNeighbors];

		/*for(int s = 0; s < 20; ++s){
			printf("%f, ", sortedDistance[s]);
		}
		printf("\n");*/

		//printf("coreDistances[%f] = %d\n", i, coreDistances[i]);
		//delete sortedDistance;
	}
	}
	/*printf("\n*************** Printing Distance Matrix ******************\n");

	int n = (rows*rows - rows)/2;
	printf("n is %d\n", n);
	for (int i = 0; i < n; i++) {
		printf("%d	|	%.1f\t\n", i,  *(distance + i));
	}*/
	/*
	printf(
			"\n*************** Printing Sorted Distance Matrix ******************\n");

	for (int i = 0; i < rows; i++) {
		printf("%.1f\t", *(coreDistances + i));

		printf("\n");
	}*/

	//printf("done");

}

void DistanceCalculator::do_euclidean(float* dataset, int numNeighbors) {

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

void DistanceCalculator::do_cosine(float* dataset, int minPoints) {
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

void DistanceCalculator::do_manhattan(float* dataset, int minPoints) {

	//absdiff(attributesOne, attributesTwo, distance);
	//reduce(distance, distance, 1, REDUCE_SUM);

	/*for (unsigned int i = 0; i < attributesOne->size() && i < attributesTwo->size(); i++) {
		distance += abs((*attributesOne)[i] - (*attributesTwo)[i]);
	}*/
}

void DistanceCalculator::do_pearson(float* dataset, int minPoints) {

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

void DistanceCalculator::do_supremum(float* dataset, int minPoints) {

	/*for (unsigned int i = 0; i < attributesOne->size() && i < attributesTwo->size(); i++) {
		double difference = abs((*attributesOne)[i] - (*attributesTwo)[i]);
		if (difference > distance)
			distance = difference;
	}*/

}

/**
 *
 */
uint DistanceCalculator::triangular(uint n){
	return (n * n + n)/2;
}

/**
 *
 */
float DistanceCalculator::getDistance(uint row, uint col){

	uint idx;
	if (row < col) {
		//printf("row > col\n");
		idx = (rows * row + col) - triangular(row + 1);

	} else if (row == col) {
		//printf("row == col\n");
		return 0;
	} else {
		//printf("row < col\n");
		idx = (rows * col + row) - triangular(col + 1);
	}

	//printf("(rows * row + col) - triangular(row + 1) : %d\n", (rows * row + col) - triangular(row + 1));
	//printf("(rows * col + row) - triangular(col + 1) : %d\n", (rows * col + row) - triangular(col + 1));

	//printf("getDistance at index %d\n", idx);
	return distance[idx];
}

}
}
