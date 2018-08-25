#ifndef COUNTINGRESULTS_H_
#define COUNTINGRESULTS_H_

#include <hdbscan/hdbscan.hpp>

#include "vocount_types.hpp"
#include "located_object.hpp"

using namespace clustering;

namespace vocount {

class CountingResults
{
public:
	CountingResults();
	virtual ~CountingResults();
	CountingResults(const CountingResults& other);
	CountingResults& operator=(const CountingResults& other);

	/************************************************************************************
	 *   GETTERS AND SETTERS
	 ************************************************************************************/
	///
	/// keypoints
	///
	vector<KeyPoint>* getKeypoints();
	void setKeypoints(vector<KeyPoint> keypoints);

	///
	/// dataset
	///
	UMat getDataset();
	void setDataset(UMat dataset);

	///
	/// distancesMap
	///
	IntDistancesMap* getDistancesMap();
	void setDistancesMap(IntDistancesMap* distancesMap);

	///
	/// selectedClustersPoints
	///
	map_kp* getSelectedClustersPoints();
	void setSelectedClustersPoints(map_kp selectedClustersPoints);

	///
	/// Stats
	///
	clustering_stats* getStats();
	void setStats(clustering_stats stats);

	///
	/// clusterMap
	///
	IntIntListMap* getClusterMap();
	void setClusterMap(IntIntListMap* clusterMap);

	///
	/// outputData
	///
	map<OutDataIndex, int32_t>* getOutputData();
	void setOutputData(map<OutDataIndex, int32_t> outputData);

	///
	/// labels
	///
	vector<int32_t>* getLabels();
	void setLabels(vector<int32_t> labels);

	///
	/// prominentLocatedObjects
	///
	vector<LocatedObject>* getProminentLocatedObjects();
	void setProminentLocatedObjects(vector<LocatedObject> prominentLocatedObjects);

	///
	/// clusterLocatedObjects
	///
	map_st* getClusterLocatedObjects();
	void setClusterLocatedObjects(map_st clusterLocatedObjects);

	///
	/// selectedClustersImages
	///
	map<String, UMat>* getSelectedClustersImages();
	void setSelectedClustersImages(map<String, UMat> selectedClustersImages);

	///
	/// validity
	///
	int32_t getValidity();
	void setValidity(int32_t val);

	///
	/// minPts
	///
	int32_t getMinPts();
	void setMinPts(int32_t val);

	/************************************************************************************
	 *   PUBLIC FUNCTIONS
	 ************************************************************************************/

	/**
	 *
	 */
	void extendLocatedObjects(UMat frame);

	/**
	 * Extract prominent
	 */
	void extractProminentLocatedObjects();

	/**
	 * Create images for the prominent located objects
	 */
	 void createLocatedObjectsImages();

	/**
	 * Generate images for each of the selected clusters
	 */
	 void generateSelectedClusterImages(UMat frame);

	 /**
	  * Generate output data
	  */
	 void generateOutputData(UMat frame, int32_t frameId, int32_t groundTruth, vector<int32_t>& roiFeatures);

	 /**
	  *
	  */
	 void addToClusterLocatedObjects(Rect2d roi, UMat& frame);

private:
	vector<KeyPoint> keypoints;
	UMat dataset;
	IntIntListMap* clusterMap;		 										/// maps labels to the keypoint indices
	//map_t roiClusterPoints;													/// cluster labels for the region of interest mapped to the roi points in the cluster
	clustering_stats stats;													/// Statistical values for the clusters
	IntDistancesMap* distancesMap;											/// Min and Max distance table for each cluster
	map_kp selectedClustersPoints;
	map<OutDataIndex, int32_t> outputData;										/// Output data
	vector<int32_t> labels;													/// hdbscan cluster labels
	vector<LocatedObject> prominentLocatedObjects;							/// Bounding boxes for the individual objects in the frame
	map_st clusterLocatedObjects;
	map<String, UMat> selectedClustersImages;								/// images with cluster by cluster keypoints drawn
	int32_t selectedFeatures = 0;
	int validity = -1;
	int minPts = 3;


/************************************************************************************
 *   PRIVATE FUNCTIONS
 ************************************************************************************/
set<int32_t> findValidROIFeature(Rect2d& roi, vector<int32_t>& roiFeatures);
};

};
#endif // COUNTINGRESULTS_H_
