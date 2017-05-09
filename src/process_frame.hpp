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
		segments,	  								/// Frame segments
		frame,
		gray,
		flow,
		dataset;
	vector<Mat> keyPointImages;
	vector<KeyPoint> keypoints, matchedKeypoints; 					/// Frame keypoints
    map<int, vector<KeyPoint> > mappedPoints;
    map<int, vector<int> > roiClusters;
	vector<int32_t> odata;
    vector<int> labels;
    vector<int32_t> cest;
	uint largest = 0;
	float lsize = 0;
	float total = 0;
	int32_t selectedSampleSize = 0;
	int32_t selectedFeatures = 0;
} framed;

typedef struct VOCOUNT{
	int step;
    map<int32_t, vector<int32_t> > stats;
    map<int32_t, vector<int32_t> > clusterEstimates;
	vector<Mat> roiDesc;
	vector<vector<KeyPoint> > roiKeypoints;
	vector<framed> frameHistory;
	Rect2d roi;

} vocount;


void display(char const* screen, const InputArray& m);

Scalar hsv_to_rgb(Scalar c);

Scalar color_mapping(int segment_id);

Mat getSegmentImage(Mat& gs, map<uint, vector<Point> >& points);

void printImage(String folder, int idx, String name, Mat img);

Mat drawKeyPoints(Mat in, vector<KeyPoint> points, Scalar colour);

void maintaintHistory(vocount& voc, framed& f);
set<int32_t> getIgnoreSegments(Rect roi, Mat segments);
void getMappedPoint(framed& f, hdbscan& scan);
void getCount(framed& f, hdbscan& scan, int ogsize);
void runSegmentation(vocount& vcount, framed& f, Ptr<GraphSegmentation> graphSegmenter, Ptr<DenseOpticalFlow> flowAlgorithm);
void mergeFlowAndImage(Mat& flow, Mat& gray, Mat& out);
Mat getDataset(vocount& vcount, framed& f, uint* ogsize);

#endif /* PROCESS_FRAME_HPP_ */
