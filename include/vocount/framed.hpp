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
    Framed(int32_t frameId, Mat& frame, Mat& descriptors, vector<KeyPoint>& keypoints, vector<int32_t> roiFeatures, Rect2d roi, int32_t groundTruth);
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
    Mat& getDescriptors();
    void setDescriptors(Mat& descriptor);

	///
	/// frame
	///
    Mat& getFrame();
    void setFrame(Mat& frame);

	///
	/// templateMatch
	///
    Mat& getTemplateMatch();

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
    vector<LocatedObject>* getCombinedLocatedObjects();

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
	CountingResults* detectDescriptorsClusters(ResultIndex idx, Mat& dataset, vector<KeyPoint>* keypoints, int32_t minPts,
												int32_t kSize, int32_t step, int32_t iterations, bool useTwo);

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
	void createResultsImages(ResultIndex idx, map<String, Mat>& selectedClustersImages);

	/**
	 * Use the colour model to filter out false positives.
	 *
	 * @param keyPoints	- the points to be used for filtering
	 */
	void combineLocatedObjets(vector<KeyPoint>* keyPoints);

	/**
	 * Use the colour model indices to detect objects in the original descriptor clusters.
	 *
	 * @param indices - indices of the colour model keypoints in the original features
	 */
	CountingResults* getColourModelObjects(vector<int32_t> *indices, int32_t minPts, int32_t iterations);

private:
    int32_t frameId = 0;
    Mat descriptors,  												/// Frame descriptors
         frame,                                                     /// Template match
         templateMatch;													/// The frame
    vector<KeyPoint> keypoints; 									/// Frame keypoints
    Rect2d roi;														/// region of interest rectangle
    vector<int32_t> roiFeatures;									/// indices of the features inside the roi
    map_r results;
    vector<LocatedObject> combinedLocatedObjects;					/// Bounding boxes for the individual objects in the frame
	int32_t groundTruth;
	int32_t kSize;

    /**********************************************************************************************************************
     *   PRIVATE FUNCTIONS
     **********************************************************************************************************************/
	CountingResults* doCluster(Mat& dataset, int32_t kSize, int32_t step, int32_t f_minPts, bool useTwo);
	void doDetectDescriptorsClusters(CountingResults* res, Mat& dataset, vector<KeyPoint>* keypoints, int32_t minPts, int32_t iterations);
	void generateSelectedClusterImages();

};
};
#endif // FRAMED_H_
