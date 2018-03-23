/*
 * process_frame.hpp
 *
 *  Created on: 3 May 2017
 *      Author: ojmakh
 */

#ifndef PROCESS_FRAME_HPP_
#define PROCESS_FRAME_HPP_
#include "vocount/print_utils.hpp"
#include "vocount/vtypes.hpp"

using namespace cv;
using namespace std;
using namespace clustering;

/**
 * 
 * 
 */
 //void help();
 
/**
 * 
 * 
 */  
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
Mat getDescriptorDataset(vector<framed>& frameHistory, int step, Mat descriptors, vector<KeyPoint> keypoints, bool includeAngle, bool includeOctave);

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
void findROIFeature(vector<KeyPoint>& keypoints, Mat& descriptors, Rect2d& roi, vector<int32_t>& roiFeatures, Mat& roiDesc, int32_t& centerFeature);

/**
 *
 */
bool processOptions(vocount& vcount, vsettings& settings, CommandLineParser& parser, VideoCapture& cap);

/**
 *
 */
void generateFinalPointClusters(vector<int32_t>& roiFeatures, results_t* res);

/**
 * Get the true count of objects from the given folder. The
 */
void getFrameTruth(String truthFolder, map<int, int>& truth);

/**
 * Use the learned colour model selection to find the colour model 
 * of the current frame.
 * 
 * @param - vcount
 * @param - f
 * @param - colourSel
 * @param - frame 
 */
void getFrameColourModel(vocount& vcount, framed& f, selection_t& colourSel, Mat& frame); 

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
selection_t detectColourModel(Mat& frame, Mat& descriptors, vector<KeyPoint>& keypoints);

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
results_t* do_cluster(results_t* res, Mat& dataset, vector<KeyPoint>& keypoints, int step, int f_minPts, bool mapDistances, bool singleRun);

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
void createBoxStructureImages(vector<box_structure>* boxStructures, map<String, Mat>* selectedClustersImages);

/**
 * Find clusters that have points inside one of the bounding boxes
 * 
 */ 
set<int32_t> extendBoxClusters(Mat& frame, results_t* res, set<int32_t>& processedClusters);

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
void getBoxStructure(results_t* res, Rect2d& rois, Mat& frame, bool extend, bool reextend);

/**
 * 
 * 
 */
void findNewROIs(Mat& frame, vector<Ptr<Tracker>>& trackers, vector<Rect2d>& newRects, vector<box_structure>* boxStructures, String trackerName);

/**
 * 
 */ 
cv::Ptr<cv::Tracker> createTrackerByName(cv::String name);

/**
 * 
 * 
 */ 
void processFrame(vocount& vcount, vsettings& settings, selection_t& colourSel, Mat& frame);

/**
 * 
 * 
 */
void finalise(vocount& vcount); 

/**
 * 
 * 
 */
results_t* clusterDescriptors(vocount& vcount, vsettings& settings, framed& f, Mat& dataset, vector<KeyPoint>& keypoints, String& keypointsFrameDir, String& keypointsDir);

/**
 * 
 * 
 */ 
void combineSelDescriptorsRawStructures(results_t* descriptorResults, results_t* seleDescriptorResults, selection_t& colourSel, vector<int32_t>& keypointStructures, set<uint>& selStructures);
  
#endif
