/*
 * print_utils.hpp
 *
 *  Created on: 1 Aug 2017
 *      Author: ojmakh
 */

#ifndef PRINT_UTILS_HPP_
#define PRINT_UTILS_HPP_

#include "vocount/vtypes.hpp"

/**
 * 
 */ 
void printEstimates(ofstream& myfile, map<OutDataIndex, int32_t>* estimates);

/**
 * 
 */ 
void printClusterEstimates(ofstream& myfile, map<OutDataIndex, int32_t>* estimates, vector<int32_t>* cest);

/***
 *
 */
void generateOutputData(vocount& vcount, Mat& frame, vector<KeyPoint>& keypoints, vector<int32_t>& roiFeatures, results_t* res, int i);

/**
 *
 */
void printStatistics(map<int32_t, map<String, double>>& stats, String& folder);

/**
 *
 */
String createDirectory(String& mainFolder, String subfolder);

/**
 *
 */
void printImages(String& folder, map<String, Mat>* images, int count);

/**
 * 
 */ 
void printImage(String folder, int idx, String name, Mat img);

/**
 * 
 */ 
void createOutputDirectories(vocount& vcount, vsettings& settings);

/**
 * 
 */ 
void printMatToFile(const Mat& mtx, String folder, String filename, int fmt);


/**
 * 
 * 
 */
void printLabelsToFile(int32_t* labels, int32_t length, String folder, int fmt);
#endif /* PRINT_UTILS_HPP_ */
