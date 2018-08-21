#ifndef VTYPES_HPP_
#define VTYPES_HPP_
/*
 * vtypes.hpp
 *
 *  Created on: 12 Feb 2018
 *      Author: ojmakh
 */

#include <hdbscan/hdbscan.hpp>
#include <opencv2/tracking.hpp>
#include <opencv/cv.hpp>
#include <vector>
#include <map>
#include <set>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;
using namespace clustering;

#define MIN_HESSIAN 100
typedef map<int32_t, vector<KeyPoint>> map_kp;

static bool VO_DEBUG = false;

enum class OutDataIndex{	
	FrameNum,
	SampleSize,
	SelectedSampleSize,
	FeatureSize,
	SelectedFeatureSize,
	NumClusters,
	ClusterSum,
	ClusterAverage,
	BoxEst,
	TruthCount,
	Validity
};

enum class ResultIndex{
	Descriptors,
	SelectedKeypoints,
	ImageSpace
};

typedef struct {
	int32_t minPts = -3;
	int32_t validity = -4;
	int32_t numClusters = 0;
    IntIntListMap* clusterKeypointIdx = NULL; 						/// maps labels to the keypoint indices
	vector<int32_t> roiFeatures;
	Mat selectedDesc;
	vector<KeyPoint> selectedKeypoints;
	set<int32_t> selectedClusters;
	vector<int32_t> oldIndices;								/// The keypoint indices before selection
} selection_t;

typedef struct _box_structure{
	Rect2d box;
	Point matchPoint;										/// Template match location
	set<int32_t> points;
	Mat img_, hsv, gray;
	MatND hist;
	double histCompare, momentsCompare;	
} box_structure;

typedef map<int32_t, vector<box_structure>> map_st;

typedef struct {
	vector<KeyPoint>* keypoints;
	Mat* dataset;
    IntIntListMap* clusterMap = NULL;		 								/// maps labels to the keypoint indices
    map<int32_t, vector<int32_t>>* roiClusterPoints;			/// cluster labels for the region of interest mapped to the roi points in the cluster
    clustering_stats stats;													/// Statistical values for the clusters
    IntDistancesMap* distancesMap = NULL;									/// Min and Max distance table for each cluster
	map_kp* finalPointClusters;
	map<OutDataIndex, int32_t>* odata;											/// Output data
    vector<int32_t>* labels;												/// hdbscan cluster labels
	vector<box_structure>* boxStructures;									/// Bounding boxes for the individual objects in the frame
	map_st* clusterStructures;
	vector<int32_t>* cest;
	map<String, Mat>* selectedClustersImages;										/// images with cluster by cluster keypoints drawn
	map<String, Mat>* leftoverClusterImages;								/// images with leftover clusters
	IntArrayList* objectClusters;
	IntArrayList* sortedAllClusters;
	double lsize = 0;
	double total = 0;
	int32_t selectedFeatures = 0;
	int ogsize;
	int validity = -1;
	int minPts = 3;
} results_t;

typedef struct FRAMED{
	int i = 0;
	Mat descriptors,  												/// Frame descriptors
		frame,														/// The frame
		gray,														/// grayscale image
		templateMatch;												/// template match image
	vector<KeyPoint> keypoints; 									/// Frame keypoints
	Rect2d roi;														/// region of interest rectangle
	vector<int32_t> roiFeatures;										/// indices of the features inside the roi
	//int32_t centerFeature = -1;											/// index of the roi central feature
	Mat roiDesc;													/// region of interest descriptors
	bool hasRoi = false;
	map<ResultIndex, results_t*> results;
} framed;

typedef struct VOCOUNT{
    int32_t frameCount = 0;
    int32_t colourMinPts;
	bool roiExtracted = false;
	Ptr<Feature2D> detector;
    Ptr<Tracker> tracker;
	Rect2d roi;
    map<int32_t, vector<int32_t> > clusterEstimates;
	vector<framed> frameHistory;
    map<int32_t, int32_t> truth;
    ofstream descriptorsClusterFile, descriptorsEstimatesFile;
    ofstream selDescClusterFile, selDescEstimatesFile;
    ofstream dfClusterFile, dfEstimatesFile;
    ofstream trainingFile, trackingFile;
	selection_t colourSel;
} vocount;

typedef struct VSETTINGS{
	String outputFolder, inputVideo, truthFolder;
    String trackerAlgorithm;    
	String colourDir, imageSpaceDir, descriptorDir, filteredDescDir,
			dfComboDir, diComboDir, dfiComboDir;
	int32_t step, rsize;											/// How many frames to use in the dataset
	bool print = false;
	bool interactive = false;
	bool selectROI = false;
	bool exit = false;
	bool dClustering = false,											/// Descriptor space clustering
		 isClustering = false,											/// Image space clustering
		 fdClustering = false,											/// Filtered descriptor clustering	
		 dfClustering = false,											/// Combine descriptor and filtered desctiptor clustering
		 diClustering = false,											/// Combine descriptor and image space clustering
		 dfiClustering = false,											/// Combine descriptor, filtered descriptor and image space clustering
		 extend = false,
		 rotationalInvariance = false;											
	bool isConsole = true;
} vsettings;

typedef struct {
	Scalar white = Scalar(255, 255, 255);
	Scalar red = Scalar(0, 0, 255);
	Scalar green = Scalar(0, 255, 0);
	Scalar blue = Scalar(255, 0, 0);
} COLOURS;



#endif /* VTYPES_HPP_ */
