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
void printData(vocount& vcount, framed& f);

/**
 *
 */
void printStatistics(map<int, map<String, double>> stats, String folder);

#endif /* PRINT_UTILS_HPP_ */
