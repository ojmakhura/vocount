#ifndef VOCOUNTER_H_
#define VOCOUNTER_H_

#include <iostream>
#include <fstream>
#include "vocount/framed.hpp"
#include "vocount/colour_model.hpp"
#include "vocount/vocutils.hpp"

namespace vocount
{

class VOCounter
{
public:
    /** Default constructor */
    VOCounter();
    /** Default destructor */
    virtual ~VOCounter();


    /**********************************************************************************************************************
     *   PUBLIC FUNCTIONS
     **********************************************************************************************************************/

    /**
     * Check settings
     */
    void processSettings();

    /**
     * Train the object's colour model
     */
    void trainColourModel(Mat& frame, vector<KeyPoint>& keypoints);

    /**
     *
     */
    void getLearnedColourModel(int32_t chosen);

    /**
     *
     */
    void chooseColourModel(Mat& frame, Mat& descriptors, vector<KeyPoint>& keypoints);

    /**
     *
     */
    void trackFrameColourModel(Mat& frame, Mat& descriptors, vector<KeyPoint>& keypoints);

    /**
     * Extracting the count estimation
     * @param frame - the current video frame
     * @param keypoints - locations of the points
     * @param descriptors - feature descriptors
     */
    void processFrame(Mat& frame, Mat& descriptors, vector<KeyPoint>& keypoints);

    /**
     * Extract the frame truth from the LMDB database
     */
    void readFrameTruth();

    /**
     * Track the initial object's ROI
     */
    void trackInitialObject(Mat& frame, Mat& descriptors, vector<KeyPoint>& keypoints, vector<int32_t>& roiFeatures);

    /**
     * Get the ground truth for the frame given by frameId
     *
     * @param frameId
     */
    int32_t getCurrentFrameGroundTruth(int32_t frameId);

    /**********************************************************************************************************************
     *   GETTERSN AND SETTERS
     **********************************************************************************************************************/
    ///
    /// frameCount
    ///
    int32_t getFrameCount();

    ///
    /// settings
    ///
    vsettings& getSettings();

    ///
    /// colourModelMaps
    ///
    map<int32_t, IntIntListMap*>* getColourModelMaps();

    ///
    /// validities
    ///
    vector<int32_t>* getValidities();

    ///
    /// colourModel
    ///
   ColourModel* getColourModel();

    ///
    /// colourModel
    ///
   vector<Framed*>* getFramedHistory();

private:
    /**********************************************************************************************************************
     *   PRIVATE MEMBERS
     **********************************************************************************************************************/
    int32_t frameCount = 0;
    bool roiExtracted = false;
    Ptr<Feature2D> detector;
    Ptr<Tracker> tracker;
    Rect2d roi;
    map_t clusterEstimates;
    vector<Framed*> framedHistory;
    vector<vector<KeyPoint>> keypointHistory;
    vector<Mat> descriptorHistory;
    map<int32_t, int32_t> truth;
    ofstream descriptorsClusterFile, descriptorsEstimatesFile;
    ofstream selDescClusterFile, selDescEstimatesFile;
    ofstream dfClusterFile, dfEstimatesFile;
    ofstream trainingFile, trackingFile;
    map<int32_t, IntIntListMap*> colourModelMaps;
    vector<int32_t> validities;
    ColourModel colourModel;
    vsettings settings;

    /**********************************************************************************************************************
     *   PRIVATE FUNCTIONS
     **********************************************************************************************************************/
    /**
     *
     *
     */
    int32_t chooseMinPts(map<uint, set<int32_t>>& numClusterMap, vector<int32_t>& validities);

    /**
     * Function to assess whether the list of minPts contains a continuous
     * series of numbers ie: each value m is preceeded by a value of m-1 and
     * preceeds a value of m+1
     *
     * @param minPts - The set of minPts that result in the same number of
     * 				   clusters
     * @return where the set is continuous or not.
     */
    bool isContinuous(set<int32_t>& minPtsList);

    /**
     * Determines whether there are more valid clustering results than
     * invalid clusters.
     *
     * @param minPtsList: the list of clusters
     * @param validities: the vector containing validities of all clustering
     * 					  results
     */
    bool isValid(set<int32_t>& minPtsList, vector<int32_t>& validities);

    /**
     * Finds the largest cluster size
     *
     * @param numClusterMap - a map of cluster sizes to clusers
     */
    int findLargestSet(map<uint, set<int32_t>>& numClusterMap);


    /**
     * Maintain a history of 10 frame processing
     *
     * @param f - Framed object to add to the history
     */
    void maintainHistory(Framed* f, Mat& descriptors, vector<KeyPoint>* keypoints);

    /**
     * Compile a descriptor dataset
     */
    Mat getDescriptorDataset(Mat& descriptors, vector<KeyPoint>& inKeypoints, vector<KeyPoint>& outKeypoints);

    /**
     *
     */
    void printResults(Framed* f, CountingResults* res, ResultIndex idx, Mat& fr, String outDir, bool print);
};
};

#endif // VOCOUNTER_H_
