/*
 * print_utils.hpp
 *
 *  Created on: 1 Aug 2017
 *      Author: ojmakh
 */

#ifndef PRINT_UTILS_HPP_
#define PRINT_UTILS_HPP_
#include "process_frame.hpp"
#include <iostream>

void printStats(String folder, map<int32_t, vector<int32_t> > stats);

void printClusterEstimates(String folder, map<int32_t, vector<int32_t> > cEstimates);


/***
 *
 */
//void printData(vocount& vcount, framed& f);
void printData(vocount& vcount, Mat& frame, vector<KeyPoint>& keypoints, vector<int>& roiFeatures, results_t& res, int i);

/**
 *
 */
void printStatistics(map<int, map<String, double>> stats, String folder);

/**
 *
 */
String createDirectory(String& mainFolder, String subfolder);

/**
 *
 */
void printImages(String folder, map<String, Mat> images, int count);

#endif /* PRINT_UTILS_HPP_ */