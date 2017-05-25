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

typedef struct FRAMED{
	int i;
	Mat descriptors,  								/// Frame descriptors
		frame,
		gray,
		dataset;
	vector<Mat> keyPointImages;
	vector<KeyPoint> keypoints; 					/// Frame keypoints
    map<int, vector<KeyPoint> > mappedKeyPoints;					/// maps labels to their keypoints
    map<int, vector<int> > mappedLabels; 							/// maps labels to their indices
    map<int, int> roiClusterCount;									/// cluster labels for the r
	vector<int32_t> odata;
    vector<int> labels;
	uint largest = 0;
	float lsize = 0;
	float total = 0;
	int32_t selectedFeatures = 0;
	vector<int32_t> cest;
	Rect2d roi;
	vector<int> roiFeatures;					/// indices of the features inside the roi
	Mat roiDesc;
	graph roiStructure;
	list<graph> objStructures;
} framed;

typedef struct VOCOUNT{
	int step, rsize;											/// How many frames to use in the dataset
	bool roiExtracted = false;
    map<int32_t, vector<int32_t> > stats;
    map<int32_t, vector<int32_t> > clusterEstimates;
	vector<framed> frameHistory;
} vocount;

typedef struct EDGE{
	float anglediff;
	float distance;
	int idxa, idxb;
} edge;

typedef struct GRAPH{
	list<edge> edgeList;
} graph;

void display(char const* screen, const InputArray& m);

Scalar hsv_to_rgb(Scalar c);

Scalar color_mapping(int segment_id);

Mat getSegmentImage(Mat& gs, map<uint, vector<Point> >& points);

void printImage(String folder, int idx, String name, Mat img);

Mat drawKeyPoints(Mat in, vector<KeyPoint> points, Scalar colour, int type);

void maintaintHistory(vocount& voc, framed& f);
set<int32_t> getIgnoreSegments(Rect roi, Mat segments);
void mapKeyPoints(vocount& vcount, framed& f, hdbscan& scan, int ogsize);
void getCount(vocount& vcount, framed& f, hdbscan& scan, int ogsize);
void runSegmentation(vocount& vcount, framed& f, Ptr<GraphSegmentation> graphSegmenter, Ptr<DenseOpticalFlow> flowAlgorithm);
void mergeFlowAndImage(Mat& flow, Mat& gray, Mat& out);
uint getDataset(vocount& vcount, framed& f);

void printStats(String folder, map<int32_t, vector<int32_t> > stats);

void printClusterEstimates(String folder, map<int32_t, vector<int32_t> > cEstimates);
void matchByBruteForce(vocount& vcount, framed& f);
vector<KeyPoint> getAllMatchedKeypoints(framed& f);
void findROIFeature(vocount& vcount, framed& f);
#endif /* PROCESS_FRAME_HPP_ */
