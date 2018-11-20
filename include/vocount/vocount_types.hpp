#ifndef VOCOUNT_TYPES_HPP_
#define VOCOUNT_TYPES_HPP_
/*
 * vobject_count.hpp
 *
 *  Created on: 12 Feb 2018
 *      Author: ojmakh
 */

#include <opencv2/tracking.hpp>
#include <opencv/cv.hpp>
#include <vector>
#include <map>
#include <set>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

#define MIN_HESSIAN 500
typedef map<int32_t, vector<KeyPoint>> map_kp;
typedef map<int32_t, vector<int32_t>> map_t;
typedef map<int32_t, vector<double>> map_d;
typedef set<int32_t> set_t;

static bool VO_DEBUG = false;

namespace vocount {

enum class VAdditions{
    SIZE,
    ANGLE,
    BOTH,
    NONE
};

enum class OutDataIndex
{
	FrameNum,
	SampleSize,
	SelectedSampleSize,
	FeatureSize,
	SelectedFeatureSize,
	NumClusters,
	ClusterSum,
	ClusterAverage,
	BoxEst,
	TruthCount,
	Validity,
	MinPts
};

enum class ResultIndex
{
	Descriptors,
	SelectedKeypoints,
	ImageSpace,
	DescriptorFilter
};


typedef struct VSETTINGS
{
	String outputFolder, inputVideo, truthFolder;
	String trackerAlgorithm;
	String colourClusteringDir, filteringDir, descriptorDir, colourModelDir,
	       combinationDir;
	int32_t step, rsize;											/// How many frames to use in the dataset
	int32_t iterations = 0;
	int32_t minPts = 3;
	int32_t x = 0,
			y = 0,
			w = 0,
			h = 0;
	bool print = false;
	bool interactive = false;
	bool selectROI = false;
	bool exit = false;
	bool descriptorClustering = false,									/// Descriptor space clustering
	     colourModelClustering = false,									/// Detect clusters in the colour model
	     combine = false,											    /// Combine descriptor and colour model clustering
	     colourModelFiltering = false,                                  /// Filter the descriptor clusters using the colour model
	     clustersOnly = false,                                          /// Create the images that show clusters
	     daisyChain = false,
	     overSegment = false;
    VAdditions additions = VAdditions::NONE;
} vsettings;

typedef struct
{
	Scalar white = Scalar(255, 255, 255);
	Scalar red = Scalar(0, 0, 255);
	Scalar green = Scalar(0, 255, 0);
	Scalar blue = Scalar(255, 0, 0);
} COLOURS;

};

#endif /* VOCOUNT_TYPES_HPP_ */
