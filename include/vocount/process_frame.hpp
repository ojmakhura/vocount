/*
 * process_frame.hpp
 *
 *  Created on: 3 May 2017
 *      Author: ojmakh
 */

#ifndef PROCESS_FRAME_HPP_
#define PROCESS_FRAME_HPP_
#include <hdbscan/hdbscan.hpp>
#include <opencv/cv.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <vector>
#include <map>
#include <set>
#include <cstdlib>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
using namespace cv::ximgproc::segmentation;
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
    IntIntListMap* roiClusterPoints = NULL;								/// cluster labels for the region of interest mapped to the roi points in the cluster
    StringDoubleMap* stats = NULL;											/// Statistical values for the clusters
    IntDoubleListMap* distancesMap = NULL;									/// Min and Max distance table for each cluster
	map_kp* finalPointClusters;
	map<String, int32_t>* odata;											/// Output data
    vector<int32_t>* labels;												/// hdbscan cluster labels
	vector<box_structure>* boxStructures;							/// Bounding boxes for the individual objects in the frame
	vector<int32_t>* cest;
	map<String, Mat>* keyPointImages;										/// images with cluster by cluster keypoints drawn
	set<int32_t>* objectClusters;
	double lsize = 0;
	double total = 0;
	int32_t selectedFeatures = 0;
	int ogsize;
	int validity = -1;
	int minPts = 3;
} results_t;

typedef struct FRAMED{
	int i;
	Mat descriptors,  												/// Frame descriptors
		frame,														/// The frame
		gray;													/// hdbscan cluster
	vector<KeyPoint> keypoints; 									/// Frame keypoints
	vector<Rect2d> rois;														/// region of interest rectangle
	vector<vector<int32_t>> roiFeatures;										/// indices of the features inside the roi
	int centerFeature = -1;											/// index of the roi central feature
	vector<Mat> roiDesc;													/// region of interest descriptors
	bool hasRoi = false;
	map<String, results_t*> results;
} framed;

typedef struct VOCOUNT{
    int frameCount = 0;
    int colourMinPts;
	String destFolder, inputPath;
	int step, rsize;											/// How many frames to use in the dataset
	bool roiExtracted = false;
	bool interactive = false;
	bool print = false;
    map<int32_t, map<String, int32_t> > stats;
    map<int32_t, vector<int32_t> > clusterEstimates;
	vector<framed> frameHistory;
    vector<int32_t> truth;
    String trackerAlgorithm;
    ofstream descriptorsClusterFile, descriptorsEstimatesFile;
    ofstream selDescClusterFile, selDescEstimatesFile;
    ofstream indexClusterFile, indexEstimatesFile;
} vocount;

results_t* initResult_t(Mat& dataset, vector<KeyPoint>& keypoints);

/**
 *
 */
void display(char const* screen, const InputArray& m);

/**
 *
 */
Scalar hsv_to_rgb(Scalar c);

/**
 *
 */
Scalar color_mapping(int segment_id);

/**
 *
 */
Mat drawKeyPoints(Mat in, vector<KeyPoint> points, Scalar colour, int type);

/**
 *
 */
void maintaintHistory(vocount& voc, framed& f);

/**
 *
 */
void generateClusterImages(Mat frame, results_t* res);

/**
 *
 */
double countPrint(IntIntListMap* roiClusterPoints, map_kp* clusterKeyPoints, vector<int32_t>* cest, int32_t& selectedFeatures, double& lsize);

/**
 *
 */
void runSegmentation(vocount& vcount, framed& f, Ptr<GraphSegmentation> graphSegmenter, Ptr<DenseOpticalFlow> flowAlgorithm);

/**
 *
 */
void mergeFlowAndImage(Mat& flow, Mat& gray, Mat& out);

/**
 *
 */
Mat getDescriptorDataset(vector<framed>& frameHistory, int step, Mat descriptors);

/**
 *
 */
void matchByBruteForce(vocount& vcount, framed& f);

/**
 *
 */
vector<KeyPoint> getAllMatchedKeypoints(framed& f);

/**
 *
 */
int32_t findROIFeature(vector<KeyPoint>& keypoints, Mat& descriptors, vector<Rect2d>& rois, vector<vector<int32_t>>& roiFeatures, vector<Mat>& roiDesc);

/**
 *
 */
bool processOptions(vocount& voc, CommandLineParser& parser, VideoCapture& cap);

/**
 *
 */
void boxStructure(map_kp* finalPointClusters, vector<KeyPoint>& keypoints, vector<Rect2d>& rois, vector<box_structure>* boxStructures, Mat& frame);

/**
 *
 */
void generateFinalPointClusters(vector<vector<int32_t>>& roiFeatures, IntIntListMap* clusterMap, IntIntListMap* roiClusterPoints, map_kp* finalPointClusters, vector<int32_t>* labels, vector<KeyPoint>* keypoints);

/**
 * Get the true count of objects from the given folder. The
 */
void getFrameTruth(String truthFolder, vector<int32_t>& truth);

/**
 *
 */
Mat getDistanceDataset(vector<int>roiIdx, Mat descriptors);

/**
 * Given the descriptors and their keypoints, find the Mat object representing the colour values
 */
Mat getColourDataset(Mat f, vector<KeyPoint> pts);

/***
 *
 */
void getSelectedKeypointsDescriptors(Mat& desc, IntArrayList* indices, Mat& out);

/**
 * Detect the optimum minPts value for colour clustering.
 */
selection_t detectColourSelectionMinPts(Mat& frame, Mat& descriptors, vector<KeyPoint>& keypoints);

/**
 * Given a list of keypoints, we find the 2D locations of the keypoints and
 * return them as an aray of size 2*points.size
 *
 * @param points - a vector of KeyPoint objects
 * @return an array of floats for the positions of the keypoints
 */
Mat getImageSpaceDataset(vector<KeyPoint> keypoint);

/**
 *
 */
results_t* do_cluster(results_t* res, Mat& dataset, vector<KeyPoint>& keypoints, int step, int f_minPts, bool mapDistances);

/**
 * Takes a hash table of cluster point indices anc creates a map
 * of cluster KeyPoint. The hash table is a GHashTable while the
 * returned map is a C++ std::map<int, vector<KeyPoint>> 
 * 
 */ 
void getKeypointMap(IntIntListMap* listMap, vector<KeyPoint>* keypoints, map_kp& mp);

/**
 * Create a vector of KeyPoint's from a lsit of keypoint indices.
 */ 
void getListKeypoints(vector<KeyPoint>& keypoints, IntArrayList* list, vector<KeyPoint>& out);

/**
 * Clean the glib hash tables and any other memory that was dynamically allocated
 * 
 */ 
void cleanResult(results_t* res);

/**
 * Given a vector of box structures, the function draws the rectangles around the identified object locations
 * 
 */ 
void createBoxStructureImages(vector<box_structure>* boxStructures, map<String, Mat>* keyPointImages);

/**
 * Find clusters that have points inside one of the bounding boxes
 * 
 */ 
void extendBoxClusters(Mat& frame, vector<box_structure>* boxStructures, vector<KeyPoint>& keypoints, map_kp* finalPointClusters, IntIntListMap* clusterMap, IntDoubleListMap* distanceMap);

/**
 * 
 * 
 */
void calculateHistogram(box_structure& bst);

/**
 * 
 * 
 */
void expandClusters(results_t* res); 

/**
 * 
 * 
 */ 
void getBoxStructure(results_t* res, vector<Rect2d>& rois, Mat& frame);
  
#endif
