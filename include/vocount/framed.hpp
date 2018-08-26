#ifndef FRAMED_H_
#define FRAMED_H_

#include <memory>
#include "counting_results.hpp"

using namespace std;
using namespace vocount;

typedef map<ResultIndex, CountingResults*> map_r;

namespace vocount
{


/**
 * This class manages the processing of one frame.
 */
class Framed
{
public:
    /** Default constructor */
    Framed();
    Framed(int32_t frameId, UMat frame, UMat descriptors, vector<KeyPoint> keypoints, vector<int32_t> roiFeatures, Rect2d roi, int32_t groundTruth);
    /** Default destructor */
    virtual ~Framed();

    /**********************************************************************************************************************
     *   PUBLIC FUNCTIONS
     **********************************************************************************************************************/

	///
	/// frameId
	///
    int32_t getFrameId();
    void setFrameId(int32_t frameId);

	///
	/// descriptors
	///
    UMat& getDescriptors();
    void setDescriptors(UMat& descriptor);

	///
	/// frame
	///
    UMat& getFrame();
    void setFrame(UMat& frame);

	///
	/// keypoints
	///
    vector<KeyPoint>* getKeypoints();

	///
	/// roi
	///
    Rect2d getROI();
    void setROI(Rect2d roi);

	///
	/// roiFeatures
	///
    vector<int32_t>* getRoiFeatures();

    ///
	/// resutls
	///
    map_r* getResults();

	///
	/// filteredLocatedObjects
	///
    vector<LocatedObject>* getFilteredLocatedObjects();

	///
	/// groundTruth
	///
	int32_t getGroundTruth();
	void setGroundTruth(int32_t groundTruth);

    /**********************************************************************************************************************
     *   PUBLIC FUNCTIONS
     **********************************************************************************************************************/
	/**
	 *
	 */
	CountingResults* detectDescriptorsClusters(ResultIndex idx, UMat& dataset, int32_t step, bool extend);

	/**
	 *
	 */
	void addResults(ResultIndex idx, CountingResults* res);

	/**
	 *
	 */
	CountingResults* getResults(ResultIndex idx);

	/**
	 *
	 */
	void createResultsImages(ResultIndex idx);

private:
    int32_t frameId = 0;
    UMat descriptors,  												/// Frame descriptors
         frame;														/// The frame
    vector<KeyPoint> keypoints; 									/// Frame keypoints
    Rect2d roi;														/// region of interest rectangle
    vector<int32_t> roiFeatures;									/// indices of the features inside the roi
    map_r results;
    vector<LocatedObject> filteredLocatedObjects;					/// Bounding boxes for the individual objects in the frame
	int32_t groundTruth;

    /**********************************************************************************************************************
     *   PRIVATE FUNCTIONS
     **********************************************************************************************************************/
	CountingResults* doCluster(UMat& dataset, int32_t step, int32_t f_minPts, bool analyse, bool singleRun);
	void generateSelectedClusterImages();
};
};
#endif // FRAMED_H_
