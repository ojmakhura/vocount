#include "vocount/colour_model.hpp"
#include <hdbscan/hdbscan.hpp>

using namespace clustering;
namespace vocount
{

ColourModel::ColourModel()
{
    minPts = -3;
    validity = -4;
    numClusters = 0;
    colourModelClusters = NULL;
}

ColourModel::~ColourModel()
{
	if(colourModelClusters != NULL)
	{
		hdbscan_destroy_cluster_map(colourModelClusters);
	}
}


/************************************************************************************
 *   GETTERS AND SETTERS
 ************************************************************************************/

///
/// minPts
///
int32_t ColourModel::getMinPts()
{
    return this->minPts;
}

void ColourModel::setMinPts(int32_t minPts)
{
    this->minPts = minPts;
}

///
/// validity
///
int32_t ColourModel::getValidity()
{
    return this->validity;
}

void ColourModel::setValidity(int32_t validity)
{
    this->validity = validity;
}

///
/// numClusters
///
int32_t ColourModel::getNumClusters()
{
    return this->numClusters;
}

void ColourModel::setNumClusters(int32_t numClusters)
{
    this->numClusters = numClusters;
}

///
/// roiFeatures
///
vector<int32_t>& ColourModel::getRoiFeatures()
{
    return this->roiFeatures;
}

void ColourModel::setRoiFeatures(vector<int32_t>& roiFeatures)
{
    this->roiFeatures = roiFeatures;
}

///
/// selectedKeypoints
///
vector<KeyPoint>& ColourModel::getSelectedKeypoints()
{
    return this->selectedKeypoints;
}

void ColourModel::setSelectedKeypoints(vector<KeyPoint>& SelectedKeypoints)
{
    this->selectedKeypoints = SelectedKeypoints;
}

///
/// selectedDesc
///
Mat& ColourModel::getSelectedDesc()
{
    return selectedDesc;
}

void ColourModel::setSelectedDesc(Mat& selectedDesc)
{
    this->selectedDesc = selectedDesc;
}

///
/// selectedClusters
///
set<int32_t>& ColourModel::getSelectedClusters()
{
    return this->selectedClusters;
}

void ColourModel::setSelectedClusters(set<int32_t>& selectedClusters)
{
    this->selectedClusters = selectedClusters;
}

///
/// colourModelClusters
///
IntIntListMap* ColourModel::getColourModelClusters()
{
    return this->colourModelClusters;
}

void ColourModel::setColourModelClusters(IntIntListMap* colourModelClusters)
{
    this->colourModelClusters = colourModelClusters;
}

///
/// selectedIndices
///
vector<int32_t>& ColourModel::getSelectedIndices()
{
    return this->selectedIndices;
}

void ColourModel::setSelectedIndices(vector<int32_t>& selectedIndices)
{
    this->selectedIndices = selectedIndices;
}

/************************************************************************************
 *   PUBLIC FUNCTIONS
 ************************************************************************************/

/**
 *
 */
void ColourModel::addToROIFeatures(int32_t idx)
{
    this->roiFeatures.push_back(idx);
}

/**
 *
 */
void ColourModel::addToSelectedKeypoints(KeyPoint keyPoint)
{
    this->selectedKeypoints.push_back(keyPoint);
}

void ColourModel::addToSelectedKeypoints(vector<KeyPoint>::iterator a, vector<KeyPoint>::iterator b)
{
    this->selectedKeypoints.insert(this->selectedKeypoints.end(), a, b);
}

/**
 *
 */
void ColourModel::addToSelectedClusters(int32_t idx)
{
    this->selectedClusters.insert(idx);
}

/**
 *
 */
void ColourModel::addToSelectedIndices(int32_t idx)
{
    this->selectedIndices.push_back(idx);
}
};
