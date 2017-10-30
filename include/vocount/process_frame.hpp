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

using namespace cv;
using namespace std;
using namespace cv::ximgproc::segmentation;
using namespace clustering;

//typedef map<int, vector<int>> map_t;
//typedef map<int, vector<double>> map_d;
typedef map<int32_t, vector<KeyPoint>> map_kp;
//typedef set<int> set_t;

typedef struct {
	int minPts = -1;
    //map_kp clusterKeyPoints;					/// maps labels to their keypoints
    IntIntListMap* clusterKeypointIdx; 						/// maps labels to the keypoint indices
	vector<int> roiFeatures;
	Mat selectedDesc;
	vector<int> selectedClusters;
} selection_t;

typedef struct _box_structure{
	Rect box;
	vector<KeyPoint> points;
	Mat img, hsv;
	Mat hist[3];
	
} box_structure;

typedef struct {
	vector<KeyPoint>* keypoints;
	Mat* dataset;
    IntIntListMap* clusterMap = NULL;		 								/// maps labels to the keypoint indices
    IntIntListMap* roiClusterPoints = NULL;								/// cluster labels for the region of interest mapped to the roi points in the cluster
    StringDoubleMap* stats = NULL;											/// Statistical values for the clusters
    IntDoubleListMap* distancesMap = NULL;									/// Min and Max distance table for each cluster
	map_kp* finalPointClusters;
	vector<int32_t>* odata;											/// Output data
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
	Rect2d roi;														/// region of interest rectangle
	vector<int> roiFeatures;										/// indices of the features inside the roi
	int centerFeature = -1;											/// index of the roi central feature
	Mat roiDesc;													/// region of interest descriptors
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
    map<int32_t, vector<int32_t> > stats;
    map<int32_t, vector<int32_t> > clusterEstimates;
	vector<framed> frameHistory;
    map<int, int> truth;
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
Mat getSegmentImage(Mat& gs, map<uint, vector<Point> >& points);

/**
 *
 */
void printImage(String folder, int idx, String name, Mat img);

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
set<int32_t> getIgnoreSegments(Rect roi, Mat segments);

/**
 *
 */
//void mapClusters(vector<int>& labels, map_kp& clusterKeyPoints, map_t& clusterKeypointIdx, vector<KeyPoint>& keypoints);

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
void findROIFeature(framed& f, selection_t& csel);

/**
 *
 */
bool processOptions(vocount& voc, CommandLineParser& parser, VideoCapture& cap);

/**
 *
 */
void boxStructure(map_kp* finalPointClusters, vector<KeyPoint>& keypoints, Rect2d& roi, vector<box_structure>* boxStructures);

/**
 *
 */
void generateFinalPointClusters(IntIntListMap* roiClusterPoints, map_kp* finalPointClusters, vector<KeyPoint>& keypoints, vector<int32_t>* labels, IntIntListMap* clusterMap);

/**
 *
 */
vector<Point2f> reduceDescriptorDimensions(Mat descriptors);

/**
 *
 */
map<int, int> splitROIPoints(framed& f, framed& f1);

/**
 * Get the true count of objects from the given folder. The
 */
map<int, int> getFrameTruth(String truthFolder);

/**
 *
 */
Mat getDistanceDataset(vector<int>roiIdx, Mat descriptors);

/**
 * map sample features to their clusters
 */
void getSampleFeatureClusters(vector<int>* roiFeatures, vector<int32_t>* labels, set<int32_t>* objectClusters, IntIntListMap* roiClusterPoints);

/**
 * Given the descriptors and their keypoints, find the Mat object representing the colour values
 */
Mat getColourDataset(Mat f, vector<KeyPoint> pts);

/***
 *
 */
Mat getSelectedKeypointsDescriptors(Mat desc, IntArrayList* indices);

/**
 * Detect the optimum minPts value for colour clustering.
 */
selection_t detectColourSelectionMinPts(Mat frame, Mat descriptors, vector<KeyPoint> keypoints);

/**
 * Given a list of keypoints, we find the 2D locations of the keypoints and
 * return them as an aray of size 2*points.size
 *
 * @param points - a vector of KeyPoint objects
 * @return an array of floats for the positions of the keypoints
 */
Mat getPointDataset(vector<KeyPoint> keypoint);

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
map_kp getKeypointMap(IntIntListMap* listMap, vector<KeyPoint>* keypoints);

/**
 * Create a vector of KeyPoint's from a lsit of keypoint indices.
 */ 
vector<KeyPoint> getListKeypoints(vector<KeyPoint> keypoints, IntArrayList* list);

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
void extendBoxClusters(vector<box_structure>* boxStructures, vector<KeyPoint>& keypoints, map_kp* finalPointClusters, IntIntListMap* clusterMap, IntDoubleListMap* distanceMap);

/**
 * 
 * 
 */
void calculateHistogram(Mat& img, Mat* hist);
  
#endif
