/*
 * process_frame.hpp
 *
 *  Created on: 3 May 2017
 *      Author: ojmakh
 */

#ifndef PROCESS_FRAME_HPP_
#define PROCESS_FRAME_HPP_
#include "hdbscan.hpp"
#include <opencv/cv.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <vector>
#include <cstdlib>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::ximgproc::segmentation;

typedef map<int, vector<int>> map_t;
typedef map<int, vector<double>> map_d;
typedef map<int, vector<KeyPoint>> map_kp;
typedef set<int> set_t;

typedef struct {
	int minPts;
	vector<int> selectedPtsIdx;
	vector<KeyPoint> selectedPts;
	Mat selectedDesc;
} selection_t;

typedef struct _box_structure{
	Rect box;
	vector<KeyPoint> points;
} box_structure;

typedef struct EDGE{
	float anglediff;
	float distance;
	int idxa, idxb;
} edge;

typedef struct GRAPH{
	vector<edge> edgeList;
} graph;

typedef struct FRAMED{
	int i, ogsize;
	Mat descriptors,  												/// Frame descriptors
		frame,														/// The frame
		gray,														/// gray scale image
		dataset;													/// hdbscan cluster
	map<String, Mat> keyPointImages;										/// images with cluster by cluster keypoints drawn
	vector<KeyPoint> keypoints; 									/// Frame keypoints
	double lsize = 0;
	double total = 0;
	int32_t selectedFeatures = 0;
	Rect2d roi;														/// region of interest rectangle
	vector<int> roiFeatures;										/// indices of the features inside the roi
	int centerFeature = -1;											/// index of the roi central feature
	Mat roiDesc;													/// region of interest descriptors
	bool hasRoi = false;

	vector<box_structure> boxStructures;							/// Bounding boxes for the individual objects in the frame
    map_kp clusterKeyPoints;					/// maps labels to their keypoints
    map_t clusterKeypointIdx; 						/// maps labels to the keypoint indices
    map_t roiClusterPoints;						/// cluster labels for the region of interest mapped to the roi points in the cluster
    map_kp finalPointClusters;					/// for clusters where there is more than one roi point in the cluster. Maps the roi point
    																/// index to all closest descriptor points indices
	vector<int32_t> odata;											/// Output data
    vector<int> labels;												/// hdbscan cluster labels

	graph roiStructure;												/// structure of the
	vector<graph> objStructures;
	vector<box_structure> boxStructures;							/// Bounding boxes for the individual objects in the frame
	vector<int32_t> cest;
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

typedef struct {
	vector<KeyPoint> keypoints;
	void* data;
	vector<box_structure> boxStructures;							/// Bounding boxes for the individual objects in the frame
    map_kp clusterKeyPoints;					/// maps labels to their keypoints
    map_t clusterKeypointIdx; 						/// maps labels to the keypoint indices
    map_t roiClusterPoints;						/// cluster labels for the region of interest mapped to the roi points in the cluster
    map_kp finalPointClusters;					/// for clusters where there is more than one roi point in the cluster. Maps the roi point
    																/// index to all closest descriptor points indices
	vector<int32_t> odata;											/// Output data
    vector<int> labels;												/// hdbscan cluster labels

	graph roiStructure;												/// structure of the
	vector<graph> objStructures;
	vector<box_structure> boxStructures;							/// Bounding boxes for the individual objects in the frame
	vector<int32_t> cest;
} results_t;

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
void mapClusters(vector<int>& labels, map_kp& clusterKeyPoints, map_t& clusterKeypointIdx, vector<KeyPoint>& keypoints);

/**
 *
 */
void getCount(framed& f);

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
uint getDataset(vocount& vcount, framed& f);

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
void boxStructure(framed& f);

/**
 *
 */
void generateFinalPointClusters(framed& f);

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
map_t mapSampleFeatureClusters(vector<int>& roiFeatures, vector<int>& labels);

/**
 * Given the descriptors and their keypoints, find the Mat object representing the colour values
 */
Mat getColourDataset(Mat f, vector<KeyPoint> pts);

/**
 * Find the minimum and maximum core distances and intra cluster distances
 */
map_d getMinMaxDistances(map_t mp, hdbscan<float>& sc, double* core);

/**
 * Get the statistics for the core distance and intra cluster distances
 */
map<String, double> getStatistics(map_d distances, double* cr, double* dr);

/**
 *
 */
int analyseStats(map<String, double> stats);

/***
 *
 */
Mat getSelected(Mat desc, vector<int> indices);

/**
 *
 */
selection_t detectColourSelectionMinPts(Mat frame, Mat descriptors, vector<KeyPoint> keypoints);

#endif /* PROCESS_FRAME_HPP_ */
