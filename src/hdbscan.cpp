/*
 * hdbscan.cpp
 *
 *  Created on: 18 May 2016
 *      Author: junior
 */

#include "hdbscan.hpp"
#include <map>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <list>
#include <set>
#include<string>

using namespace clustering;
using namespace clustering::distance;
namespace clustering {

string getWarningMessage() {
	string message =
			"----------------------------------------------- WARNING -----------------------------------------------\n";

	message +=
			"(infinite) for some data objects, either due to replicates in the data (not a set) or due to numerical\n";
	message +=
			"roundings. This does not affect the construction of the density-based clustering hierarchy, but\n";
	message +=
			"it affects the computation of cluster stability by means of relative excess of mass. For this reason,\n";
	message +=
			"the post-processing routine to extract a flat partition containing the most stable clusters may\n";
	message +=
			"produce unexpected results. It may be advisable to increase the value of MinPts and/or M_clSize.\n";
	message +=
			"-------------------------------------------------------------------------------------------------------";

	return message;
}

HDBSCAN::HDBSCAN(calculator cal, uint minPoints, uint minClusterSize) {
	this->minPoints = minPoints;
	this->minClusterSize = minClusterSize;
	distanceFunction.setCalculator(cal);
	selfEdges = true;
	constraints = NULL;
	clusterLabels = NULL;
	//dataSet = NULL;
	mst = NULL;
	clusters = NULL; //new vector<Cluster*>();
	outlierScores = NULL;
	//coreDistances = NULL;
	hierarchy = NULL;
	numPoints = 0;
}
HDBSCAN::HDBSCAN(float* dataSet, int rows, int cols, calculator cal,
		uint minPoints, uint minClusterSize) {

	this->minClusterSize = minClusterSize;
	this->minPoints = minPoints;
	distanceFunction.setCalculator(cal);
	selfEdges = true;
	constraints = new vector<Constraint*>();
	clusterLabels = NULL;
	mst = NULL;
	clusters = NULL; //new vector<Cluster*>();;
	outlierScores = NULL;
	hierarchy = NULL;
	calculateCoreDistances(dataSet, rows, cols);
	numPoints = rows;
	clusterLabels = new vector<int>(numPoints, 0);
	//printf("distances.cols(), distances.rows() numPoints : %d %d %d\n", cols, rows, numPoints);
}

HDBSCAN::HDBSCAN(vector<vector<float> > dataset, calculator cal, uint minPoints, uint minClusterSize){

	this->minClusterSize = minClusterSize;
	this->minPoints = minPoints;
	//distanceFunction.setCalculator(cal);
	selfEdges = true;
	constraints = new vector<Constraint*>();
	clusterLabels = NULL;
	mst = NULL;
	clusters = NULL; //new vector<Cluster*>();;
	outlierScores = NULL;
	hierarchy = NULL;
	calculateCoreDistances(dataset);
	numPoints = dataset[0].size();
	clusterLabels = new vector<int>(numPoints, 0);

}

HDBSCAN::HDBSCAN(vector<vector<Point> > contours, Mat frame, Mat flow, calculator cal, uint minPoints, uint minClusterSize){
	this->minClusterSize = minClusterSize;
	this->minPoints = minPoints;
	distanceFunction.setCalculator(cal);
	selfEdges = true;
	constraints = NULL;
	mst = NULL;
	clusters = NULL; //new vector<Cluster*>();;
	outlierScores = NULL;
	hierarchy = NULL;
	//printf("cvCalculateCoreDistances(contours, frame, flow)\n");
	cvCalculateCoreDistances(contours, frame, flow);
	//printf("cvCalculateCoreDistances(contours, frame, flow) ================================\n");
	//calculateCoreDistances(dataSet, rows, cols);
	numPoints = 0;

	for(uint i = 0; i < contours.size(); ++i){
		for(uint j = 0; j < contours[i].size(); ++j){
			++numPoints;
		}
	}
	clusterLabels = new vector<int>(numPoints, 0);
}


HDBSCAN::HDBSCAN(vector<Point> contour, vector<Mat> dataset, calculator cal, uint minPoints, uint minClusterSize){
	this->minClusterSize = minClusterSize;
	this->minPoints = minPoints;
	distanceFunction.setCalculator(cal);
	selfEdges = true;
	constraints = NULL;
	mst = NULL;
	clusters = NULL; // new vector<Cluster*>();
	outlierScores = NULL;
	hierarchy = NULL;
	numPoints = contour.size();
	clusterLabels = new vector<int>(numPoints, 0);
	cout << " calculating core distances " << endl;
	calculateCoreDistances(contour, dataset);

}

HDBSCAN::HDBSCAN(vector<Mat > dataset, calculator cal, uint minPoints, uint minClusterSize, bool indexed){
	this->minClusterSize = minClusterSize;
	this->minPoints = minPoints;
	distanceFunction.setCalculator(cal);
	selfEdges = true;
	constraints = NULL;
	mst = NULL;
	clusters = NULL; // new vector<Cluster*>();
	outlierScores = NULL;
	hierarchy = NULL;
	//printf("cvCalculateCoreDistances(contours, frame, flow)\n");
	cvCalculateCoreDistances(dataset, indexed);
	//printf("cvCalculateCoreDistances(contours, frame, flow) ================================\n");
	//calculateCoreDistances(dataSet, rows, cols);
	numPoints = dataset[0].rows * dataset[0].cols;
	clusterLabels = new vector<int>(numPoints, 0);
}

HDBSCAN::HDBSCAN(Mat& dataset, calculator cal, uint minPoints, uint minClusterSize){
	this->minClusterSize = minClusterSize;
	this->minPoints = minPoints;
	distanceFunction.setCalculator(cal);
	selfEdges = true;
	constraints = NULL;
	mst = NULL;
	clusters = NULL; // new vector<Cluster*>();
	outlierScores = NULL;
	hierarchy = NULL;
	//printf("cvCalculateCoreDistances(contours, frame, flow)\n");
	//printf("cvCalculateCoreDistances(contours, frame, flow) ================================\n");
	//calculateCoreDistances(dataSet, rows, cols);
	numPoints = dataset.rows;
	clusterLabels = new vector<int>(numPoints, 0);
	distanceFunction.cvComputeDistance(dataset, minPoints-1);
}

HDBSCAN::HDBSCAN(string fileName, calculator cal, uint minPoints,
		uint minClusterSize) {
	this->minPoints = minPoints;
	this->minClusterSize = minClusterSize;
	distanceFunction.setCalculator(cal);
	selfEdges = true;
	constraints = NULL;
	mst = NULL;
	clusters = NULL; //new vector<Cluster*>();;
	outlierScores = NULL;
	hierarchy = NULL;
	int rows = 0;
	int cols = 0;
	float* dataset = readInDataSet(fileName, &rows, &cols);
	calculateCoreDistances(dataset, rows, cols);
	numPoints = rows;
	clusterLabels = new vector<int>(numPoints, 0);
}

HDBSCAN::HDBSCAN(string dataFileName, string constraintFileName, calculator cal,
		uint minPoints, uint minClusterSize) {
	this->minPoints = minPoints;
	this->minClusterSize = minClusterSize;
	distanceFunction.setCalculator(cal);
	selfEdges = true;
	constraints = NULL;
	clusterLabels = NULL;
	mst = NULL;
	clusters = NULL;
	outlierScores = NULL;
	hierarchy = NULL;
	//readInDataSet(dataFileName);
	readInConstraints(constraintFileName);
	//calculateCoreDistances();
	//numPoints = dataSet.rows();
	//distances = MatrixXf(numPoints, numPoints);
	clusterLabels = new vector<int>(numPoints, 0);
}

HDBSCAN::~HDBSCAN() {

	if (constraints != NULL) {

		for(uint i = 0; i < constraints->size(); ++i){
			delete (*constraints)[i];
		}

		delete constraints;
	}

	/*if (distanceFunction != NULL) {
		delete distanceFunction;
	}*/

	if (clusterLabels != NULL) {
		delete clusterLabels;
	}

	if (mst != NULL) {
		delete mst;
	}

	if (clusters != NULL) {

		for(uint i = 0; i < clusters->size(); ++i){
			delete (*clusters)[i];
		}

		delete clusters;
	}

	if (outlierScores != NULL) {

		for(uint i = 0; i < outlierScores->size(); ++i){
			delete (*outlierScores)[i];
		}

		delete outlierScores;
	}

	if(hierarchy != NULL){

		for(map<long, vector<int>* >::iterator it = hierarchy->begin(); it != hierarchy->end(); ++it){
			//vector<int>* v = it->second;

			delete it->second;
		}

		delete hierarchy;
	}
}

/**
 * Reads in the input data set from the file given, assuming the delimiter separates attributes
 * for each data point, and each point is given on a separate line.  Error messages are printed
 * if any part of the input is improperly formatted.
 * @param fileName The path to the input file
 * @param delimiter A regular expression that separates the attributes of each point
 * @return A vector<float>[] where index [i][j] indicates the jth attribute of data point i
 * @throws IOException If any errors occur opening or reading from the file
 */
float* HDBSCAN::readInDataSet(string fileName, int* rows, int* cols) {

	float* dataSet;
	//dataSet = imread(fileName);
	//dataSet = new vector<vector<float> >();
	std::ifstream inFile(fileName);
	string item;
	int l = 0;

	while (inFile) {
		if (!getline(inFile, item)) {
			break;
		}

		istringstream ss(item);
		vector<float> line;

		(*cols) = 0;
		while (ss) {
			string s;
			if (!getline(ss, s, ',')) {
				break;
			}
			//line.push_back(atof(s.c_str()));
			++(*cols);
			++l;
		}
		//dataSet->push_back(line);
		++(*rows);
		++l;
	}
	//printf("dataset will have length %d of cols = %d and rows = %d\n", (*cols) * (*rows), *cols, *rows);
	std::ifstream inFile2(fileName);
	int i = 0;
	dataSet = new float[(*cols) * (*rows)];
	while (inFile2) {
			if (!getline(inFile2, item)) {
				break;
			}

			cout << item << endl;

			istringstream ss(item);
			vector<float> line;

			//(*cols) = 0;
			//j = 0;
			while (ss) {
				string s;
				if (!getline(ss, s, ',')) {
					break;
				}
				float f = atof(s.c_str());
				dataSet[i] = f;
				//++(*cols);
				++i;
			}
			//dataSet->push_back(line);
			//++(*rows);
			//++i;
		}

	return dataSet;
}

/**
 * Reads in constraints from the file given, assuming the delimiter separates the points involved
 * in the constraint and the type of the constraint, and each constraint is given on a separate
 * line.  Error messages are printed if any part of the input is improperly formatted.
 * @param fileName The path to the input file
 * @param delimiter A regular expression that separates the points and type of each constraint
 * @return An vector of Constraints
 * @throws IOException If any errors occur opening or reading from the file
 */
void HDBSCAN::readInConstraints(string fileName) {

	constraints = new vector<Constraint*>();
	std::ifstream inFile(fileName);
	string item;
	/*while (getline(inFile, item, ',')) {
		CONSTRAINT_TYPE type;
		int pointA, pointB;

		pointA = atoi(item.c_str());
		getline(inFile, item, ',');
		pointB = atoi(item.c_str());
		getline(inFile, item, '\n');
		if (item == MUST_LINK_TAG) {
			type = MUST_LINK;
		} else if (item == CANNOT_LINK_TAG) {
			type = CANNOT_LINK;
		}

		constraints->push_back(new Constraint(pointA, pointB, type));
	}*/

}

/**
 * Calculates the core distances for each point in the data set, given some value for k.
 * @param dataSet A vector<vector<float> > where index [i][j] indicates the jth attribute of data point i
 * @param k Each point's core distance will be it's distance to the kth nearest neighbor
 * @param distanceFunction A DistanceCalculator to compute distances between points
 * @return An array of core distances
 */
void HDBSCAN::calculateCoreDistances(float* dataSet, int rows, int cols) {

	uint size = rows;

	int numNeighbors = minPoints - 1;

	if (minPoints == 1 && size < minPoints) {
		return;
	}

	distanceFunction.computeDistance(dataSet, rows, cols, numNeighbors);

}

void HDBSCAN::calculateCoreDistances(vector<Point> contour, vector<Mat> dataset) {
	int numNeighbors = minPoints - 1;
	distanceFunction.cvComputeEuclidean(contour, dataset, numNeighbors, true);

}

void HDBSCAN::calculateCoreDistances(vector<vector<float> >& dataset){
	int numNeighbors = minPoints - 1;
	distanceFunction.computeDistance(dataset, numNeighbors);

}

void HDBSCAN::cvCalculateCoreDistances(vector<vector<Point> > contours, Mat frame, Mat flow) {

	/*uint size = rows;



	if (minPoints == 1 && size < minPoints) {
		return;
	}*/
	int numNeighbors = minPoints - 1;
	distanceFunction.cvComputeEuclidean(contours, flow, frame, numNeighbors);

}

void HDBSCAN::cvCalculateCoreDistances(vector<Mat> dataset, bool indexed) {

	/*uint size = rows;



	if (minPoints == 1 && size < minPoints) {
		return;
	}*/
	int numNeighbors = minPoints - 1;
	distanceFunction.cvComputeDistance(dataset, numNeighbors, indexed);

}

/**
 * Constructs the minimum spanning tree of mutual reachability distances for the data set, given
 * the core distances for each point.
 * @param dataSet A vector<vector<float> > where index [i][j] indicates the jth attribute of data point i
 * @param coreDistances An array of core distances for each data point
 * @param selfEdges If each point should have an edge to itself with weight equal to core distance
 * @param distanceFunction A DistanceCalculator to compute distances between points
 */

void HDBSCAN::constructMST(Mat& src) {

	int nb_edges = 0;
	Mat dataset = src.clone();
	numPoints = dataset.rows * dataset.cols;
	clusterLabels = new vector<int>(numPoints, 0);
	int nb_channels = dataset.channels();

	int selfEdgeCapacity = 0;
	uint size = dataset.rows * dataset.cols;
	if (selfEdges) {
		//printf("Self edges set to true\n");
		selfEdgeCapacity = size;
	}

	//One bit is set (true) for each attached point, or unset (false) for unattached points:
	//bool attachedPoints[selfEdgeCapacity] = { };
	//The MST is expanded starting with the last point in the data set:
	//unsigned int currentPoint = size - 1;
	//int numAttachedPoints = 1;
	//attachedPoints[size - 1] = true;

	//Each point has a current neighbor point in the tree, and a current nearest distance:
	vector<int>* nearestMRDNeighbors = new vector<int>(size - 1 + selfEdgeCapacity);
	vector<float>* weights = new vector<float>(size - 1 + selfEdgeCapacity, numeric_limits<float>::max());

	//Create an array for vertices in the tree that each point attached to:
	vector<int>* otherVertexIndices = new vector<int>(size - 1 + selfEdgeCapacity);

	for (int i = 0; i < dataset.rows; i++) {
		const float* p = dataset.ptr<float>(i);

		for (int j = 0; j < dataset.cols; j++) {

			//Mat ds(Size(3,3), CV_32FC1, numeric_limits<float>::max());
			//vector<float> dsv(9, numeric_limits<float>::max());

			int from = i * dataset.rows + j;
			int to;
			int r = 0, c = 0, min_r = i, min_c = j;
			float min_d = numeric_limits<float>::max();
			float coreDistance = 0;
			min_d = numeric_limits<float>::max();

			for(int k = -1; k <= 1; ++k){
				for(int l = -1; l <= 1; ++l){

					float distance = 0;
					r = i+k;
					c = j+l;
					const float* p2 = dataset.ptr<float>(r);

					if(r >= 0 && c >= 0 && r < dataset.rows && c < dataset.cols && !(r == i && c == j)){
						//cout << "calculating distance " << endl;
						distance = 0;

						// calculate the distance between (i,j) and (k,l)
						for (int channel = 0; channel < nb_channels; channel++) {
							//printf("channel %d : (%f, %f)\n", channel, p[j * nb_channels + channel], p2[c * nb_channels + channel]);
							distance += pow(p[j * nb_channels + channel] - p2[c * nb_channels + channel], 2);
						}

						distance = sqrt(distance);

						if(distance > coreDistance){
							coreDistance = distance;
						}

						if(distance < min_d){

							//int from = i * dataset.rows + j;
							to = r * dataset.rows + c;
							min_d = distance;

							// If this edge does not exist
							if((*nearestMRDNeighbors)[to] != from){
								min_r = r;
								min_c = c;
							}
						}
					} else if(r == i && c == j){
						distance = 0;
					}

					//printf("(%d, %d) : (%d, %d) -> (%f, %f, %f)\n", i, j, r, c, distance, coreDistance, min_d);
					//ds.at<float>(r,c) = distance;
					//++c;
				}
				//++r;
			}
			//cout << from << " " << to << " " << min_d << endl;
			//printf("(%d, %d), (%d, %d) -> %f\n", i, j, min_r, min_c, min_d);
			//printf("size (otherVertexIndices, nearestMRDNeighbors, weights) = (%d, %d, %d)\n", otherVertexIndices->size(), nearestMRDNeighbors->size(), weights->size());
			(*otherVertexIndices)[from] = from;
			(*nearestMRDNeighbors)[from] = to;
			(*weights)[from] = coreDistance;

			if (selfEdges) {
				(*otherVertexIndices)[from + size - 1] = from;
				(*nearestMRDNeighbors)[from + size - 1] = from;
				(*weights)[from + size - 1] = coreDistance;
			}
		}
	}
	//mst = new UndirectedGraph(size, nearestMRDNeighbors, otherVertexIndices, weights);
	//printf("Dimensions: (%d, %d) -> \n", dataset.rows, dataset.cols);
	//cout << "Printing graph" << endl;
	//mst.print();
	//cout << "Done" << endl;
}

void HDBSCAN::constructMST() {

	//float* distances = distanceFunction.getDistance();
	float* coreDistances = distanceFunction.getCoreDistances();

	int selfEdgeCapacity = 0;
	uint size = numPoints;
	if (selfEdges){
		//printf("Self edges set to true\n");
		selfEdgeCapacity = size;
	}

	//One bit is set (true) for each attached point, or unset (false) for unattached points:
	bool attachedPoints[selfEdgeCapacity] = {};
	//The MST is expanded starting with the last point in the data set:
	unsigned int currentPoint = size - 1;
	//int numAttachedPoints = 1;
	attachedPoints[size - 1] = true;

	//Each point has a current neighbor point in the tree, and a current nearest distance:
	vector<int>* nearestMRDNeighbors = new vector<int>(size - 1 + selfEdgeCapacity);
	vector<float>* nearestMRDDistances = new vector<float>(size - 1 + selfEdgeCapacity, numeric_limits<float>::max());

	//Create an array for vertices in the tree that each point attached to:
	vector<int>* otherVertexIndices = new vector<int>(size - 1 + selfEdgeCapacity);

	//Continue attaching points to the MST until all points are attached:
	for(uint numAttachedPoints = 1; numAttachedPoints < size; numAttachedPoints++){
	//while (numAttachedPoints < size) {
		int nearestMRDPoint = -1;
		float nearestMRDDistance = numeric_limits<float>::max();

		//Iterate through all unattached points, updating distances using the current point:
		//parallel_for(size_t(0), n, [=](size_t i) {Foo(a[i]);});
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
		for (unsigned int neighbor = 0; neighbor < size; neighbor++) {
			if (currentPoint == neighbor) {
				continue;
			}

			if (attachedPoints[neighbor]) {
				continue;
			}

			float mutualReachabiltiyDistance = distanceFunction.getDistance(neighbor, currentPoint);

			if (coreDistances[currentPoint] > mutualReachabiltiyDistance) {
				mutualReachabiltiyDistance = coreDistances[currentPoint];
			}

			if (coreDistances[neighbor] > mutualReachabiltiyDistance) {
				mutualReachabiltiyDistance = coreDistances[neighbor];
			}

			if (mutualReachabiltiyDistance < (*nearestMRDDistances)[neighbor]) {
				(*nearestMRDDistances)[neighbor] = mutualReachabiltiyDistance;
				(*nearestMRDNeighbors)[neighbor] = currentPoint;
			}

			//Check if the unattached point being updated is the closest to the tree:
			if ((*nearestMRDDistances)[neighbor] <= nearestMRDDistance) {
				nearestMRDDistance = (*nearestMRDDistances)[neighbor];
				nearestMRDPoint = neighbor;
			}

		}

		//Attach the closest point found in this iteration to the tree:
		attachedPoints[nearestMRDPoint] = true;
		(*otherVertexIndices)[numAttachedPoints] = numAttachedPoints;
		//numAttachedPoints++;
		currentPoint = nearestMRDPoint;

	}


	//If necessary, attach self edges:
	if (selfEdges) {
		size_t n = size * 2 - 1;

		//parallel_for(size_t(0), n, [=](size_t i) {
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
		for (uint i = size - 1; i < size * 2 - 1; i++) {
			int vertex = i - (size - 1);
			(*nearestMRDNeighbors)[i] = vertex;
			(*otherVertexIndices)[i] = vertex;
			(*nearestMRDDistances)[i] = coreDistances[vertex];
			//printf("At %d coreDistances[%d] = %f\n", i, vertex, coreDistances[vertex]);
		}
	}

	mst = new UndirectedGraph(size, *nearestMRDNeighbors, *otherVertexIndices, *nearestMRDDistances);
	//mst->print();

}

/**
 * Propagates constraint satisfaction, stability, and lowest child death level from each child
 * cluster to each parent cluster in the tree.  This method must be called before calling
 * findProminentClusters() or calculateOutlierScores().
 * @param clusters A list of Clusters forming a cluster tree
 * @return true if there are any clusters with infinite stability, false otherwise
 */
bool HDBSCAN::propagateTree() {

	map<int, Cluster*> clustersToExamine;

	vector<bool> addedToExaminationList(clusters->size());
	bool infiniteStability = false;

	//Find all leaf clusters in the cluster tree:
//#ifndef USE_OPENMP
//#pragma omp parallel for
//#endif
	for(uint i = 0; i < clusters->size(); i++){
		Cluster* cluster = (*clusters)[i];
		if (cluster != NULL && !cluster->hasKids()) {
			//if()
			clustersToExamine.insert(
					pair<int, Cluster*>(cluster->getLabel(), cluster));
			addedToExaminationList[cluster->getLabel()] = true;
		}
	}

	/*for (vector<Cluster*>::iterator itr = clusters->begin(); itr != clusters->end(); ++itr) {
		Cluster* cluster = *itr;
		if (cluster != NULL && !cluster->hasKids()) {
			//if()
			clustersToExamine.insert(
					pair<int, Cluster*>(cluster->getLabel(), cluster));
			addedToExaminationList[cluster->getLabel()] = true;
		}

	}*/

	//Iterate through every cluster, propagating stability from children to parents:
	while (!clustersToExamine.empty()) {
		map<int, Cluster*>::reverse_iterator itr = clustersToExamine.rbegin();
		pair<int, Cluster*> p = *itr;
		Cluster* currentCluster = p.second;
		clustersToExamine.erase(p.first);
		currentCluster->propagate();

		if (currentCluster->getStability() == numeric_limits<float>::max())
			infiniteStability = true;

		if (currentCluster->getParent() != NULL) {
			Cluster* parent = currentCluster->getParent();

			if (!addedToExaminationList[parent->getLabel()]) {
				clustersToExamine.insert(
						pair<int, Cluster*>(parent->getLabel(), parent));
				addedToExaminationList[parent->getLabel()] = true;
			}
		}
	}


	if (infiniteStability)
		printf("%s\n", getWarningMessage().c_str());
	return infiniteStability;
}

/**
 * Produces the outlier score for each point in the data set, and returns a sorted list of outlier
 * scores.  propagateTree() must be called before calling this method.
 * @param clusters A list of Clusters forming a cluster tree which has already been propagated
 * @param pointNoiseLevels A vector<float> with the levels at which each point became noise
 * @param pointLastClusters An vector<int> with the last label each point had before becoming noise
 * @param coreDistances An array of core distances for each data point
 * @param outlierScoresOutputFile The path to the outlier scores output file
 * @param delimiter The delimiter for the output file
 * @param infiniteStability true if there are any clusters with infinite stability, false otherwise
 */
void HDBSCAN::calculateOutlierScores(vector<float>* pointNoiseLevels,
		vector<int>* pointLastClusters, bool infiniteStability) {

	float* coreDistances = distanceFunction.getCoreDistances();
	int numPoints = pointNoiseLevels->size();
	//printf("Creating outlierScores\n");
	outlierScores = new vector<OutlierScore*>();
	outlierScores->reserve(numPoints);
	//printf("Created outlierScores\n");
	//Iterate through each point, calculating its outlier score:

	/*int i = 0;

	for(vector<Cluster*>::iterator it = clusters->begin(); it != clusters->end(); ++it){
		Cluster* c = *it;
		float epsilon_max = c->getPropagatedLowestChildDeathLevel();
		//printf("hey too\n");
		float epsilon = (*pointNoiseLevels)[i];

		float score = 0;
		if (epsilon != 0) {
			score = 1 - (epsilon_max / epsilon);
		}
		//printf("3333333333 %d of %d outlierScores size %d\n", i, numPoints, outlierScores->size(), coreDistances);
		outlierScores->push_back(new OutlierScore(score, coreDistances[i], i));
		//printf("Finishes createing and pushing outlier\n");
		++i;
	}*/

	for (int i = 0; i < numPoints; i++) {
		printf("hey %d\n", pointLastClusters->size());

		int tmp = (*pointLastClusters)[i];

		//printf("heyxxxxxxxxxxx\n");
		Cluster* c = (*clusters)[tmp];
		//printf("fffffffffffffffffffffffffff %d\n", c);
		float epsilon_max =
				c->getPropagatedLowestChildDeathLevel();
		//printf("hey too\n");
		float epsilon = (*pointNoiseLevels)[i];

		float score = 0;
		if (epsilon != 0) {
			score = 1 - (epsilon_max / epsilon);
		}
		//printf("3333333333 %d of %d outlierScores size %d\n", i, numPoints, outlierScores->size(), coreDistances);
		outlierScores->push_back(
				new OutlierScore(score, coreDistances[i], i));
		//printf("Finishes createing and pushing outlier\n");
	}
	//printf("outlierScores for loop complete\n");
	//Sort the outlier scores:
	std::sort(outlierScores->begin(), outlierScores->end());
	//printf("outlierScores sorted\n");
}

// ------------------------------ PRIVATE METHODS ------------------------------

/**
 * Removes the set of points from their parent Cluster, and creates a new Cluster, provided the
 * clusterId is not 0 (noise).
 * @param points The set of points to be in the new Cluster
 * @param clusterLabels An array of cluster labels, which will be modified
 * @param parentCluster The parent Cluster of the new Cluster being created
 * @param clusterLabel The label of the new Cluster
 * @param edgeWeight The edge weight at which to remove the points from their previous Cluster
 * @return The new Cluster, or NULL if the clusterId was 0
 */
Cluster* HDBSCAN::createNewCluster(set<int>* points, vector<int>* clusterLabels,
		Cluster* parentCluster, int clusterLabel, float edgeWeight) {

	for (set<int>::iterator it = points->begin(); it != points->end(); ++it) {
		int idx = *it;
		(*clusterLabels)[idx] = clusterLabel;
	}

	parentCluster->detachPoints(points->size(), edgeWeight);

	if (clusterLabel != 0) {

		int s = (int) points->size();
		return new Cluster(clusterLabel, parentCluster, edgeWeight, s);
	}

	else {
		parentCluster->addPointsToVirtualChildCluster(*points);
		return NULL;
	}
}

/**
 * Calculates the number of constraints satisfied by the new clusters and virtual children of the
 * parents of the new clusters.
 * @param newClusterLabels Labels of new clusters
 * @param clusters An vector of clusters
 * @param constraints An vector of constraints
 * @param clusterLabels an array of current cluster labels for points
 */
void HDBSCAN::calculateNumConstraintsSatisfied(set<int> newClusterLabels,
		vector<int> currentClusterLabels) {

	if (constraints == NULL || constraints->size() == 0) {
		return;
	}

	bool contains;
	vector<Cluster*> parents;

	for (set<int>::iterator it = newClusterLabels.begin();
			it != newClusterLabels.end(); ++it) {
		Cluster* parent = (*clusters)[*it]->getParent();

		contains = find(parents.begin(), parents.end(), parent)
				!= parents.end();
		if (!contains)
			parents.push_back(parent);
	}

	for (vector<Constraint*>::iterator it = constraints->begin();
			it != constraints->end(); ++it) {
		Constraint* constraint = *it;
		int labelA = currentClusterLabels[constraint->getPointA()];
		int labelB = currentClusterLabels[constraint->getPointB()];

		if (constraint->getType() == MUST_LINK && labelA == labelB) {
			if (newClusterLabels.find(labelA) != newClusterLabels.end()) {
				(*clusters)[labelA]->addConstraintsSatisfied(2);
			}
		} else if (constraint->getType() == CANNOT_LINK
				&& (labelA != labelB || labelA == 0)) {

			contains = newClusterLabels.find(labelA) != newClusterLabels.end();
			if (labelA != 0 && contains) {
				(*clusters)[labelA]->addConstraintsSatisfied(1);
			}

			contains = newClusterLabels.find(labelB) != newClusterLabels.end();
			if (labelB != 0 && contains) {
				(*clusters)[labelB]->addConstraintsSatisfied(1);
			}

			if (labelA == 0) {
				for (vector<Cluster*>::iterator it = parents.begin();
						it != parents.end(); ++it) {
					Cluster* parent = *it;
					if (parent->virtualChildClusterContaintsPoint(
							constraint->getPointA())) {
						parent->addVirtualChildConstraintsSatisfied(1);
						break;
					}
				}
			}

			if (labelB == 0) {
				for (vector<Cluster*>::iterator it = parents.begin();
						it != parents.end(); ++it) {
					Cluster* parent = *it;
					if (parent->virtualChildClusterContaintsPoint(
							constraint->getPointB())) {
						parent->addVirtualChildConstraintsSatisfied(1);
						break;
					}
				}
			}
		}
	}

	for (vector<Cluster*>::iterator it = parents.begin(); it != parents.end();
			++it) {
		Cluster* parent = *it;
		parent->releaseVirtualChildCluster();
	}
}

vector<int>& HDBSCAN::getClusterLabels() {
	return *clusterLabels;
}

map<int, float>& HDBSCAN::getClusterStabilities(){
	return clusterStabilities;
}

/*
MatrixXf HDBSCAN::getDataSet() {
	return dataSet;
}
*/

vector<Cluster*>& HDBSCAN::getClusters() {
	return *clusters;
}

/**
 * Computes the hierarchy and cluster tree from the minimum spanning tree, writing both to file,
 * and returns the cluster tree.  Additionally, the level at which each point becomes noise is
 * computed.  Note that the minimum spanning tree may also have self edges (meaning it is not
 * a true MST).
 * @param mst A minimum spanning tree which has been sorted by edge weight in descending order
 * @param minClusterSize The minimum number of points which a cluster needs to be a valid cluster
 * @param compactHierarchy Indicates if hierarchy should include all levels or only levels at
 * which clusters first appear
 * @param constraints An optional ArrayList of Constraints to calculate cluster constraint satisfaction
 * @param hierarchyOutputFile The path to the hierarchy output file
 * @param treeOutputFile The path to the cluster tree output file
 * @param delimiter The delimiter to be used while writing both files
 * @param pointNoiseLevels A float[] to be filled with the levels at which each point becomes noise
 * @param pointLastClusters An int[] to be filled with the last label each point had before becoming noise
 * @return The cluster tree
 * @throws IOException If any errors occur opening or writing to the files
 */
void HDBSCAN::computeHierarchyAndClusterTree(bool compactHierarchy,
		vector<float>* pointNoiseLevels, vector<int>* pointLastClusters) {

	//mst->print();
	int lineCount = 0; // Indicates the number of lines written into
								// hierarchyFile.

	//The current edge being removed from the MST:
	int currentEdgeIndex = mst->getNumEdges() - 1;
	hierarchy = new map<long, vector<int>*>();

	int nextClusterLabel = 2;
	bool nextLevelSignificant = true;
	clusters = new vector<Cluster*>();
	//The previous and current cluster numbers of each point in the data set:
	vector<int>* previousClusterLabels = new vector<int>(mst->getNumVertices(), 1);
	vector<int>* currentClusterLabels = new vector<int>(mst->getNumVertices(), 1);

	//A list of clusters in the cluster tree, with the 0th cluster (noise) null:
	clusters->push_back(NULL);
	clusters->push_back(new Cluster(1, NULL, NAN, mst->getNumVertices()));


	if (constraints != NULL && constraints->size() > 0) {//Calculate number of constraints satisfied for cluster 1:
		set<int> clusterOne;
		clusterOne.insert(1);
		calculateNumConstraintsSatisfied(clusterOne, *currentClusterLabels);
		//delete clusterOne;
	}

	//

	//Sets for the clusters and vertices that are affected by the edge(s) being removed:
	set<int>* affectedClusterLabels = new set<int>();
	set<int>* affectedVertices = new set<int>();

	while (currentEdgeIndex >= 0) {

		float currentEdgeWeight = mst->getEdgeWeightAtIndex(currentEdgeIndex);
		vector<Cluster*>* newClusters = new vector<Cluster*>();
		//Remove all edges tied with the current edge weight, and store relevant clusters and vertices:
		while (currentEdgeIndex >= 0 && mst->getEdgeWeightAtIndex(currentEdgeIndex) == currentEdgeWeight) {

			int firstVertex = mst->getFirstVertexAtIndex(currentEdgeIndex);
			int secondVertex = mst->getSecondVertexAtIndex(currentEdgeIndex);

			mst->removeEdge(firstVertex, secondVertex);
			if ((*currentClusterLabels)[firstVertex] == 0) {
				currentEdgeIndex--;
				continue;
			}

			affectedVertices->insert(firstVertex);
			affectedVertices->insert(secondVertex);

			affectedClusterLabels->insert((*currentClusterLabels)[firstVertex]);

			currentEdgeIndex--;
		}

		//Check each cluster affected for a possible split:
		while (!affectedClusterLabels->empty()) {
			set<int>::reverse_iterator it = affectedClusterLabels->rbegin();
			int examinedClusterLabel = *it;
			affectedClusterLabels->erase(examinedClusterLabel);
			set<int> examinedVertices;// = new set<int>();

			//Get all affected vertices that are members of the cluster currently being examined:
			for (set<int>::iterator itr = affectedVertices->begin();
					itr != affectedVertices->end(); ++itr) {
				int n = *itr;
				if ((*currentClusterLabels)[n] == examinedClusterLabel) {
					examinedVertices.insert(n);
					affectedVertices->erase(n);
				}
			}

			set<int>* firstChildCluster = NULL;
			vector<int>* unexploredFirstChildClusterPoints = NULL;
			int numChildClusters = 0;

			/* Check if the cluster has split or shrunk by exploring the graph from each affected
			 * vertex.  If there are two or more valid child clusters (each has >= minClusterSize
			 * points), the cluster has split.
			 * Note that firstChildCluster will only be fully explored if there is a cluster
			 * split, otherwise, only spurious components are fully explored, in order to label
			 * them noise.
			 */
			while (!examinedVertices.empty()) {

				//TODO Clean up this
				set<int> constructingSubCluster;// = new set<int>();
				vector<int> unexploredSubClusterPoints;// = new vector<int>();

				bool anyEdges = false;
				bool incrementedChildCount = false;

				set<int>::reverse_iterator itr = affectedClusterLabels->rbegin();
				itr = examinedVertices.rbegin();
				int rootVertex = *itr;
				std::pair<std::set<int>::iterator, bool> p =
						constructingSubCluster.insert(rootVertex);

				unexploredSubClusterPoints.push_back(rootVertex);
				examinedVertices.erase(rootVertex);

				//Explore this potential child cluster as long as there are unexplored points:
				while (!unexploredSubClusterPoints.empty()) {
					int vertexToExplore = *(unexploredSubClusterPoints.begin());
					unexploredSubClusterPoints.erase(
							unexploredSubClusterPoints.begin());
					vector<int>* v = mst->getEdgeListForVertex(vertexToExplore);
					for (vector<int>::iterator itr = v->begin();
							itr != v->end(); ++itr) {
						int neighbor = *itr;
						anyEdges = true;

						p = constructingSubCluster.insert(neighbor);
						if (p.second) {
							unexploredSubClusterPoints.push_back(neighbor);
							examinedVertices.erase(neighbor);
						}
					}

					//Check if this potential child cluster is a valid cluster:
					if (!incrementedChildCount
							&& constructingSubCluster.size() >= minClusterSize
							&& anyEdges) {
						incrementedChildCount = true;
						numChildClusters++;

						//If this is the first valid child cluster, stop exploring it:
						if (firstChildCluster == NULL) {
							firstChildCluster = new set<int>(constructingSubCluster.begin(), constructingSubCluster.end());
							unexploredFirstChildClusterPoints =	new vector<int>(unexploredSubClusterPoints.begin(), unexploredSubClusterPoints.end());
							break;
						}
					}
				}

				//If there could be a split, and this child cluster is valid:
				if (numChildClusters >= 2
						&& constructingSubCluster.size() >= minClusterSize
						&& anyEdges) {
					//Check this child cluster is not equal to the unexplored first child cluster:
					it = firstChildCluster->rbegin();
					int firstChildClusterMember = *it;
					if (constructingSubCluster.find(firstChildClusterMember)
							!= constructingSubCluster.end()) {
						numChildClusters--;
					}

					//Otherwise, create a new cluster:
					else {

						Cluster* newCluster = createNewCluster(
								&constructingSubCluster, currentClusterLabels,
								(*clusters)[examinedClusterLabel],
								nextClusterLabel, currentEdgeWeight);
						//printf("Otherwise, create a new cluster: %d of label %d\n", newCluster, newCluster->getLabel());
						newClusters->push_back(newCluster);
						clusters->push_back(newCluster);

						nextClusterLabel++;
					}
				}

				//If this child cluster is not valid cluster, assign it to noise:
				else if (constructingSubCluster.size() < minClusterSize
						|| !anyEdges) {

					createNewCluster(
							&constructingSubCluster, currentClusterLabels,
							(*clusters)[examinedClusterLabel], 0,
							currentEdgeWeight);

					for (set<int>::iterator itr = constructingSubCluster.begin(); itr != constructingSubCluster.end(); ++itr) {
						int point = *itr;
						(*pointNoiseLevels)[point] = currentEdgeWeight;
						(*pointLastClusters)[point] = examinedClusterLabel;
					}
				}

				//delete constructingSubCluster;
				//delete unexploredSubClusterPoints;
			}

			//Finish exploring and cluster the first child cluster if there was a split and it was not already clustered:
			if (numChildClusters >= 2
					&& (*currentClusterLabels)[*(firstChildCluster->begin())]
							== examinedClusterLabel) {

				while (!unexploredFirstChildClusterPoints->empty()) {
					vector<int>::iterator it =
							unexploredFirstChildClusterPoints->begin();
					int vertexToExplore = *it;
					unexploredFirstChildClusterPoints->erase(
							unexploredFirstChildClusterPoints->begin());
					vector<int>* v = mst->getEdgeListForVertex(vertexToExplore);

					for (vector<int>::iterator itr = v->begin();
							itr != v->end(); ++itr) {
						int neighbor = *itr;
						std::pair<std::set<int>::iterator, bool> p =
								firstChildCluster->insert(neighbor);
						if (p.second) {
							unexploredFirstChildClusterPoints->push_back(
									neighbor);
						}

					}
				}

				Cluster* newCluster = createNewCluster(firstChildCluster,
						currentClusterLabels, (*clusters)[examinedClusterLabel],
						nextClusterLabel, currentEdgeWeight);
				newClusters->push_back(newCluster);
				//printf("Finish exploring %d\n", newCluster);
				clusters->push_back(newCluster);
				nextClusterLabel++;
			}
			//delete examinedVertices;
			delete firstChildCluster;
			delete unexploredFirstChildClusterPoints;
			// TODO clean up newClusters
		}

		/*if (newClusters->empty()) {
			printf("compactHierarchy: %d, nextLevelSignificant: %d, newClusters: %d\n ", compactHierarchy, nextLevelSignificant, newClusters);
		} else {
			printf("compactHierarchy: %d, nextLevelSignificant: %d, newClusters.size(): %ld\n", compactHierarchy, nextLevelSignificant, newClusters->size());
		}*/

		//Write out the current level of the hierarchy:
		if (!compactHierarchy || nextLevelSignificant
				|| !newClusters->empty()) {

			lineCount++;

			//TODO: Clean the memory allocation with new
			hierarchy->insert(pair<long, vector<int>*>(lineCount, new vector<int>(previousClusterLabels->begin(), previousClusterLabels->end())));
		}

		// Assign file offsets and calculate the number of constraints
					// satisfied:
		set<int>* newClusterLabels = new set<int>();
		for (vector<Cluster*>::iterator itr = newClusters->begin(); itr != newClusters->end(); ++itr) {
			Cluster* newCluster = *itr;

			newCluster->setOffset(lineCount);
			newClusterLabels->insert(newCluster->getLabel());
		}

		if (!newClusterLabels->empty()){
			calculateNumConstraintsSatisfied(*newClusterLabels, *currentClusterLabels);
		}

		for (uint i = 0; i < previousClusterLabels->size(); i++) {
			(*previousClusterLabels)[i] = (*currentClusterLabels)[i];
		}

		if (newClusters->empty())
			nextLevelSignificant = false;
		else
			nextLevelSignificant = true;

		delete newClusterLabels;
		delete newClusters;
	}
	// TODO: have to figure out how to clean this
	vector<int>* labels = new vector<int>();
	// Write out the final level of the hierarchy (all points noise):
	for (uint i = 0; i < previousClusterLabels->size() - 1; i++) {
		labels->push_back(0);
	}
	labels->push_back(0);
	hierarchy->insert(pair<long, vector<int>*>(0, labels));
	lineCount++;

	//mst->print();

	delete previousClusterLabels;
	delete currentClusterLabels;
	delete affectedClusterLabels;
	delete affectedVertices;
}

void HDBSCAN::run(bool calcDistances) {
	int start = clock();
	/*if (calcDistances) {
		calculateCoreDistances(dataSet, rows, cols);
	}*/
	int stop = clock();

	//cout << "calculateCoreDistances time : " << (stop-start)/double(CLOCKS_PER_SEC)*1000 << endl;

	start = clock();
	//Calculate minimum spanning tree:
	//cout << "construct MST" << endl;
	constructMST();
	//cout << "constructed MST" << endl;
	stop = clock();

	//cout << "constructMST time : " << (stop-start)/double(CLOCKS_PER_SEC)*1000 << endl;

	start = clock();
	//cout << "quicksortByEdgeWeight" << endl;
	mst->quicksortByEdgeWeight();
	//cout << "quicksortByEdgeWeight done" << endl;
	stop = clock();

	//cout << "quicksortByEdgeWeight time : " << (stop-start)/double(CLOCKS_PER_SEC)*1000 << endl;

	start = clock();

	//int numPoints = coreDistances->size();


	// Remove references to unneeded objects:
	//vector<vector<float> >().swap(*dataSet);
	//dataSet = NULL;

	vector<float>* pointNoiseLevels = new vector<float>(numPoints);
	vector<int>* pointLastClusters = new vector<int>(numPoints);
	//Compute hierarchy and cluster tree:

	start = clock();
	//cout << "computeHierarchyAndClusterTree" << endl;
	computeHierarchyAndClusterTree(false, pointNoiseLevels, pointLastClusters);
	//cout << "computeHierarchyAndClusterTree done" << endl;
	//Remove references to unneeded objects:
	mst = NULL;
	//Propagate clusters:
	stop = clock();

	//cout << "computeHierarchyAndClusterTree time : " << (stop-start)/double(CLOCKS_PER_SEC)*1000 << endl;

	start = clock();
	//cout << "infiniteStability" << endl;
	bool infiniteStability = propagateTree();
	//cout << "infiniteStability done" << endl;

	stop = clock();

	//cout << "propagateTree time : " << (stop-start)/double(CLOCKS_PER_SEC)*1000 << endl;

	start = clock();
	//Compute final flat partitioning:
	//cout << "findProminentClusters" << endl;
	findProminentClusters(infiniteStability);
	//cout << "findProminentClusters done" << endl;
	stop = clock();

	//cout << "findProminentClusters time : " << (stop-start)/double(CLOCKS_PER_SEC)*1000 << endl;

	start = clock();
	//Compute outlier scores for each point:
	//cout << "calculateOutlierScores" << endl;
	//calculateOutlierScores(pointNoiseLevels, pointLastClusters,
			//infiniteStability);
	//cout << "calculateOutlierScores done" << endl;

	stop = clock();

	//cout << "calculateOutlierScores time : " << (stop-start)/double(CLOCKS_PER_SEC)*1000 << endl;

	start = clock();;

	//cout << endl;

	delete pointNoiseLevels;
	delete pointLastClusters;
}

/**
 * Produces a flat clustering result using constraint satisfaction and cluster stability, and
 * returns an array of labels.  propagateTree() must be called before calling this method.
 * @param clusters A list of Clusters forming a cluster tree which has already been propagated
 * @param hierarchyFile The path to the hierarchy input file
 * @param flatOutputFile The path to the flat clustering output file
 * @param delimiter The delimiter for both files
 * @param numPoints The number of points in the original data set
 * @param infiniteStability true if there are any clusters with infinite stability, false otherwise
 */
void HDBSCAN::findProminentClusters(bool infiniteStability) {

	vector<Cluster*>* solution = (*clusters)[1]->getPropagatedDescendants();
	clusterStabilities.insert(pair<int, float>(0, 0.0f));
	/*for(vector<Cluster*>::iterator it = solution->begin(); it != solution->end(); ++it){
		Cluster
		if()
	}*/

	if(clusterLabels == NULL){
		clusterLabels = new vector<int>(numPoints, 0);
	}
	map<long, vector<int> > significant;
	set<int> toInspect;
//#pragma omp parallel for
	for (vector<Cluster*>::iterator itr = solution->begin(); itr != solution->end(); ++itr) {
		Cluster* cluster = *itr;
		if (cluster != NULL) {
			vector<int>* clusterList = &significant[cluster->getOffset()];
			if (significant.find(cluster->getOffset()) == significant.end()) {
				clusterList = new vector<int>();
				significant.insert(pair<long, vector<int> >(cluster->getOffset(), *clusterList));
				//printf("inserting into significant");
				//clusterStabilities.insert(pair<int, float>(cluster->getLabel(), cluster->getStability()));
			}

			clusterList->push_back(cluster->getLabel());

		}
	}

	//printf("clusters size: %d and significant size: %d\n\n[", clusters->size(), significant.size());

	while(!significant.empty()){
		pair<long, vector<int> > p = *(significant.begin());
		significant.erase(significant.begin());
		vector<int> clusterList = p.second;
		long offset = p.first;
		hierarchy->size();
		vector<int>* hpSecond = (*hierarchy)[offset+1];

		for(uint i = 0; i < hpSecond->size(); i++){
			int label = (*hpSecond)[i];
			vector<int>::iterator it = find(clusterList.begin(), clusterList.end(), label);
			if(it != clusterList.end()){
				(*clusterLabels)[i] = label;

			}

		}
	}

	for(uint d = 0; d < clusterLabels->size(); ++d){
		for (vector<Cluster*>::iterator itr = solution->begin(); itr != solution->end(); ++itr) {
			Cluster* cluster = *itr;
			if(cluster->getLabel() == (*clusterLabels)[d]){
				clusterStabilities.insert(pair<int, float>(cluster->getLabel(), cluster->getStability()));
			}
		}
	}

}

bool HDBSCAN::compareClusters(Cluster* one, Cluster* two){

	return one == two;
}

void HDBSCAN::clean(){

}

}
/* namespace clustering */
