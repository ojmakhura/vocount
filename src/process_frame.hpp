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
#include <vector>
#include <cstdlib>
#include <iostream>

using namespace cv;
using namespace std;



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
    map<int32_t, vector<int32_t> > stats;
    map<int32_t, vector<int32_t> > clusterEstimates;
    map<int, vector<KeyPoint> > mappedPoints;
    map<int, vector<int> > roiClusters;
    vector<int> labels;
	uint largest = 0;
	float lsize = 0;
	float total = 0;
} framed;

typedef struct VOCOUNT{
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

#endif /* PROCESS_FRAME_HPP_ */
