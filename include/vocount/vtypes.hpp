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

typedef map<int32_t, vector<KeyPoint>> map_kp;

static String frameNum = "Frame No.";
static String sampleSize = "Sample Size";
static String selectedSampleSize = "Selected Sample";
static String featureSize = "Feature Size";
static String selectedFeatureSize = "Selected Features";
static String numClusters = "# Clusters";
static String clusterSum = "Cluster Sum";
static String clusterAverage = "Cluster Avg.";
static String boxEst = "Box Est.";
static 	String truthCount = "Actual";
static String validityStr = "Validity";

typedef struct {
	int minPts = -1;
    IntIntListMap* clusterKeypointIdx; 						/// maps labels to the keypoint indices
	vector<vector<int32_t>> roiFeatures;
	Mat selectedDesc;
	vector<KeyPoint> selectedKeypoints;
	set<int32_t> selectedClusters;
} selection_t;

typedef struct _box_structure{
	Rect box;
	vector<KeyPoint> points;
	Mat img_, hsv;
	MatND hist;
	double histCompare, momentsCompare;	
} box_structure;

typedef struct {
	vector<KeyPoint>* keypoints;
	Mat* dataset;
    IntIntListMap* clusterMap = NULL;		 								/// maps labels to the keypoint indices
    IntIntListMap* roiClusterPoints = NULL;									/// cluster labels for the region of interest mapped to the roi points in the cluster
    clustering_stats stats;													/// Statistical values for the clusters
    IntDistancesMap* distancesMap = NULL;									/// Min and Max distance table for each cluster
	map_kp* finalPointClusters;
	map<String, int32_t>* odata;											/// Output data
    vector<int32_t>* labels;												/// hdbscan cluster labels
	vector<box_structure>* boxStructures;									/// Bounding boxes for the individual objects in the frame
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
		gray;													/// hdbscan cluster
	vector<KeyPoint> keypoints; 									/// Frame keypoints
	vector<Rect2d> rois;														/// region of interest rectangle
	vector<vector<int32_t>> roiFeatures;										/// indices of the features inside the roi
	vector<int32_t> centerFeatures;											/// index of the roi central feature
	vector<Mat> roiDesc;													/// region of interest descriptors
	bool hasRoi = false;
	map<String, results_t*> results;
} framed;

typedef struct VOCOUNT{
    int frameCount = 0;
    int colourMinPts;
	bool roiExtracted = false;
	Ptr<Feature2D> detector;
    vector<Ptr<Tracker>> trackers;
	vector<Rect2d> rois;
    map<int32_t, map<String, int32_t> > stats;
    map<int32_t, vector<int32_t> > clusterEstimates;
	vector<framed> frameHistory;
    map<int32_t, int32_t> truth;
    ofstream descriptorsClusterFile, descriptorsEstimatesFile;
    ofstream selDescClusterFile, selDescEstimatesFile;
    ofstream indexClusterFile, indexEstimatesFile;
} vocount;

typedef struct VSETTINGS{
	String outputFolder, inputVideo, truthFolder;
    String trackerAlgorithm;    
	String colourDir, imageSpaceDir, descriptorDir, filteredDescDir;
	int step, rsize;											/// How many frames to use in the dataset
	bool print = false;
	bool interactive = false;
	bool selectROI = false;
	bool exit = false;
	bool dClustering = false,											/// Descriptor space clustering
		 isClustering = false,											/// Image space clustering
		 fdClustering = false,											/// Filtered descriptor clustering	
		 dfClustering = false,											/// Combine descriptor and filtered desctiptor clustering
		 diClustering = false,											/// Combine descriptor and image space clustering
		 dfiClustering = false;											/// Combine descriptor, filtered descriptor and image space clustering
	
} vsettings;


#endif /* VTYPES_HPP_ */
