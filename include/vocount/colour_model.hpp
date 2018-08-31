#ifndef COLOURMODEL_H_
#define COLOURMODEL_H_

#include <listlib/utils.h>
#include <opencv/cv.hpp>
#include <vector>
#include <map>
#include <set>

using namespace cv;
using namespace std;

namespace vocount
{

class ColourModel
{
public:
    ColourModel();
    virtual ~ColourModel();

	/************************************************************************************
     *   GETTERS AND SETTERS
     ************************************************************************************/
    ///
    /// minPts
    ///
    int32_t getMinPts();
    void setMinPts(int32_t minPts);

    ///
    /// validity
    ///
    int32_t getValidity();
    void setValidity(int32_t validity);

    ///
    /// numClusters
    ///
    int32_t getNumClusters();
    void setNumClusters(int32_t numClusters);

    ///
    /// roiFeatures
    ///
    vector<int32_t>* getRoiFeatures();
    void setRoiFeatures(vector<int32_t> roiFeatures);

    ///
    /// selectedKeypoints
    ///
    vector<KeyPoint>* getSelectedKeypoints();
    void setSelectedKeypoints(vector<KeyPoint> selectedKeypoints);

    ///
    /// selectedDesc
    ///
    Mat& getSelectedDesc();
    void setSelectedDesc(Mat& selectedDesc);

    ///
    /// selectedClusters
    ///
    set<int32_t>* getSelectedClusters();
    void setSelectedClusters(set<int32_t> selectedClusters);

    ///
    /// oldIndices
    ///
    vector<int32_t>* getOldIndices();
    void setOldIndices(vector<int32_t> oldIndices);

    ///
    /// colourModelClusters
    ///
    IntIntListMap* getColourModelClusters();
    void setColourModelClusters(IntIntListMap* colourModelClusters);

	/************************************************************************************
     *   PUBLIC FUNCTIONS
     ************************************************************************************/

    /**
     *
     */
    void addToROIFeatures(int32_t idx);

    /**
     *
     */
    void addToSelectedKeypoints(KeyPoint keyPoint);


    void addToSelectedKeypoints(vector<KeyPoint>::iterator a, vector<KeyPoint>::iterator b);

    /**
     *
     */
    void addToSelectedClusters(int32_t idx);

    /**
     *
     */
    void addToOldIndices(int32_t idx);

private:
    int32_t minPts;
    int32_t validity;
    int32_t numClusters;
    IntIntListMap* colourModelClusters;
    vector<int32_t> roiFeatures;
    vector<KeyPoint> selectedKeypoints;
    Mat selectedDesc;
    set<int32_t> selectedClusters;
    vector<int32_t> oldIndices;

	/************************************************************************************
     *   PRIVATE FUNCTIONS
     ************************************************************************************/
};
};

#endif // COLOURMODEL_H_
