#include "vocount/counting_results.hpp"
#include "vocount/vocutils.hpp"
#include "vocount/voprinter.hpp"

namespace vocount
{

CountingResults::CountingResults()
{
    this->clusterMap = NULL;
    this->distancesMap = NULL;
}

CountingResults::~CountingResults()
{
    if(this->clusterMap != NULL)
    {
        hdbscan_destroy_cluster_map(this->clusterMap);
        this->clusterMap = NULL;
    }

    if(this->distancesMap != NULL)
    {
        hdbscan_destroy_distance_map(this->distancesMap);
        this->distancesMap = NULL;
    }
}

CountingResults::CountingResults(const CountingResults& other)
{
    //copy ctor
}

CountingResults& CountingResults::operator=(const CountingResults& rhs)
{
    if (this == &rhs)
        return *this; // handle self assignment
    //assignment operator
    return *this;
}

/************************************************************************************
 *   GETTERS AND SETTERS
 ************************************************************************************/
///
/// keypoints
///
vector<KeyPoint>& CountingResults::getKeypoints()
{
    return this->keypoints;
}

void CountingResults::setKeypoints(vector<KeyPoint>& keypoints)
{
    this->keypoints.clear();
    this->keypoints.insert(this->keypoints.begin(), keypoints.begin(), keypoints.end());
}

///
/// dataset
///
Mat& CountingResults::getDataset()
{
    return this->dataset;
}

void CountingResults::setDataset(Mat& dataset)
{
    this->dataset = dataset;
}

///
/// distancesMap
///
IntDistancesMap* CountingResults::getDistancesMap()
{
    return this->distancesMap;
}

void CountingResults::setDistancesMap(IntDistancesMap* distancesMap)
{
    this->distancesMap = distancesMap;
}

///
/// selectedClustersPoints
///
map_kp& CountingResults::getSelectedClustersPoints()
{
    return this->selectedClustersPoints;
}

void CountingResults::setSelectedClustersPoints(map_kp& selectedClustersPoints)
{
    this->selectedClustersPoints = selectedClustersPoints;
}

///
/// Stats
///
clustering_stats& CountingResults::getStats()
{
    return this->stats;
}

void CountingResults::setStats(clustering_stats& stats)
{
    this->stats = stats;
}

///
/// clusterMap
///
IntIntListMap* CountingResults::getClusterMap()
{
    return this->clusterMap;
}

void CountingResults::setClusterMap(IntIntListMap* clusterMap)
{
    this->clusterMap = clusterMap;
}

///
/// outputData
///
map<OutDataIndex, int32_t>& CountingResults::getOutputData()
{
    return this->outputData;
}

void CountingResults::setOutputData(map<OutDataIndex, int32_t>& outputData)
{
    this->outputData = outputData;
}

///
/// labels
///
vector<int32_t>& CountingResults::getLabels()
{
    return this->labels;
}

void CountingResults::setLabels(vector<int32_t>& labels)
{
    this->labels = labels;
}

///
/// prominentLocatedObjects
///
vector<LocatedObject>& CountingResults::getProminentLocatedObjects()
{
    return this->prominentLocatedObjects;
}

void CountingResults::setProminentLocatedObjects(vector<LocatedObject>& prominentLocatedObjects)
{
    this->prominentLocatedObjects = prominentLocatedObjects;
}

///
/// minPts
///
double CountingResults::getRunningTime()
{
    return this->runningTime;
}

void CountingResults::setRunningTime(double runningTime)
{
    this->runningTime = runningTime;
}

///
/// clusterLocatedObjects
///
map_st& CountingResults::getClusterLocatedObjects()
{
    return this->clusterLocatedObjects;
}

void CountingResults::addToClusterLocatedObjects(VRoi roi, Mat& frame)
{
    vector<int32_t> validObjFeatures;
    VOCUtils::findValidROIFeatures(this->keypoints, roi.getBox(), validObjFeatures, this->labels);

    // sort the valid features by how close to the center they are
    VOCUtils::sortByDistanceFromCenter(roi.getBox(), validObjFeatures, keypoints);

    LocatedObject mbs; /// Create a box structure based on roi
    mbs.setBoundingBox(roi);
    mbs.setMatchTo(roi);
    Mat f_image = frame(roi.getBox());
    mbs.setBoxImage(f_image);
    Mat h = VOCUtils::calculateHistogram(mbs.getBoxImage());
    mbs.setHistogram(h);
    mbs.setHistogramCompare(compareHist(mbs.getHistogram(), mbs.getHistogram(), cv::HISTCMP_CORREL ));
    mbs.setMomentsCompare (matchShapes(mbs.getBoxGray(), mbs.getBoxGray(), CONTOURS_MATCH_I3, 0));

    /// generate box structures based on the valid object points
    for(size_t i = 0; i < validObjFeatures.size(); i++)
    {
        int32_t key = labels.at(validObjFeatures[i]);

        // Check if the cluster has not already been processed
        if(clusterLocatedObjects.find(key) != clusterLocatedObjects.end())
        {
            continue;
        }

        vector<LocatedObject>& availableOjects= clusterLocatedObjects[key];
        ArrayList* l1 = array_list_init(12, sizeof(int32_t *), int_ptr_compare);
        
        int r = hashtable_lookup(clusterMap, &key, l1);

        if(r == 0)
        {
            continue;
        }

        KeyPoint f_point = keypoints.at(validObjFeatures[i]);

        /// Each point in the cluster should be inside a LocatedObject
        mbs.addToPoints(validObjFeatures[i]);
        int32_t* data = (int32_t *)l1->data;

        for(int32_t j = 0; j < l1->size; j++)
        {
            KeyPoint& t_point = keypoints.at(data[j]);
            if(mbs.getBoundingBox().getBox().contains(t_point.pt))  // if the point is inside mbs, add it to mbs' points
            {
                mbs.addToPoints(data[j]);
            }
            else    // else create a new mbs for it
            {

                LocatedObject newObject;
                if(LocatedObject::createNewLocatedObject(f_point, t_point, mbs, newObject, frame))
                {
                    newObject.addToPoints(data[j]);
                    newObject.setMatchTo(mbs.getBoundingBox());
                    LocatedObject::addLocatedObject(availableOjects, newObject);
                }
            }
        }
        availableOjects.push_back(mbs); // TODO: some cluster match objects might not be getting updated
    }
}


void CountingResults::setClusterLocatedObjects(map_st& clusterLocatedObjects)
{
    this->clusterLocatedObjects = clusterLocatedObjects;
}

///
/// validity
///
int32_t CountingResults::getValidity()
{
    return this->validity;
}

void CountingResults::setValidity(int32_t val)
{
    this->validity = val;
}

///
/// minPts
///
int32_t CountingResults::getMinPts()
{
    return this->minPts;
}

void CountingResults::setMinPts(int32_t val)
{
    this->minPts = val;
}

/************************************************************************************
 *   PUBLIC FUNCTIONS
 ************************************************************************************/

/**
 *
 */
void CountingResults::generateSelectedClusterImages(Mat& frame, map<String, Mat>& selectedClustersImages, OutputType outputType)
{
    COLOURS colours;
    vector<KeyPoint> kp;
    for(map_st::iterator it = clusterLocatedObjects.begin(); it != clusterLocatedObjects.end(); ++it)
    {
        int32_t key = it->first;

        ArrayList* l1 = array_list_init(12, sizeof(int32_t *), int_ptr_compare);
        
        int r = hashtable_lookup(clusterMap, &key, l1);

        if(r == 0)
        {
            continue;
        }

        vector<KeyPoint>& kps = this->selectedClustersPoints[key];
        VOCUtils::getListKeypoints(this->keypoints, l1, kps);

        this->selectedFeatures += kps.size();

        if(outputType == OutputType::ALL)
        {
            Mat kimg = VOCUtils::drawKeyPoints(frame, kps, colours.red, -1);
            vector<LocatedObject>& rects = it->second;
            VRoi tmp;
            for(uint i = 0; i < rects.size() - 1; i++)
            {
                RNG rng(12345);
                Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                                    rng.uniform(0, 255));

                tmp = rects.at(i).getBoundingBox();
                //rectangle(kimg, tmp.getBox(), value, 2, 8, 0);

                line(kimg, tmp.getP1(), tmp.getP2(), value, 2);
                line(kimg, tmp.getP2(), tmp.getP3(), value, 2);
                line(kimg, tmp.getP3(), tmp.getP4(), value, 2);
                line(kimg, tmp.getP4(), tmp.getP1(), value, 2);
                
            }

            tmp = rects.at(rects.size() - 1).getBoundingBox();
            //rectangle(kimg, tmp.getBox(), colours.red, 2, 8, 0);
            line(kimg, tmp.getP1(), tmp.getP2(), colours.red, 2);
            line(kimg, tmp.getP2(), tmp.getP3(), colours.red, 2);
            line(kimg, tmp.getP3(), tmp.getP4(), colours.red, 2);
            line(kimg, tmp.getP4(), tmp.getP1(), colours.red, 2);
            
            String ss = "img_keypoints-";
            string s = to_string(key);
            ss += s.c_str();


            distance_values dv ;
            
            int r = hashtable_lookup(this->distancesMap, &key, &dv);

            if(r == 0)
            {
                continue;
            }

            ss += "-";
            ss += to_string((int)dv.dr_confidence);
            ss += "-";
            ss += to_string((int)dv.dr_confidence);
            selectedClustersImages[ss] = kimg.clone();
        }
        
        this->selectedNumClusters += 1;
        kp.insert(kp.end(), kps.begin(), kps.end());
    }

    Mat mm = VOCUtils::drawKeyPoints(frame, kp, colours.red, -1);

    String ss = "img_allkps";
    selectedClustersImages[ss] = mm;
    this->selectedNumClusters += 1;
}

/**
 *
 */
void CountingResults::createLocatedObjectsImages(map<String, Mat>& selectedClustersImages)
{
    String ss = "img_bounds";
    Mat img_bounds = selectedClustersImages["img_allkps"].clone();

    for (size_t i = 0; i < this->prominentLocatedObjects.size(); i++)
    {
        LocatedObject& b = this->prominentLocatedObjects[i];
        RNG rng(12345);
        Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                              rng.uniform(0, 255));
        //rectangle(img_bounds, b.getBoundingBox().getBox(), value, 2, 8, 0);

        line(img_bounds, b.getBoundingBox().getP1(), b.getBoundingBox().getP2(), value, 2);
        line(img_bounds, b.getBoundingBox().getP2(), b.getBoundingBox().getP3(), value, 2);
        line(img_bounds, b.getBoundingBox().getP3(), b.getBoundingBox().getP4(), value, 2);
        line(img_bounds, b.getBoundingBox().getP4(), b.getBoundingBox().getP1(), value, 2);
    }
    selectedClustersImages[ss] = img_bounds;
    this->selectedNumClusters += 1;
}

void CountingResults::generateOutputData(int32_t frameId, int32_t groundTruth, vector<int32_t>& roiFeatures)
{
    int selSampleSize = 0;
    if(this->clusterLocatedObjects.size() > 0)
    {
        outputData[OutDataIndex::SampleSize] = roiFeatures.size();
        outputData[OutDataIndex::BoxEst] = prominentLocatedObjects.size();
    }
    else
    {
        outputData[OutDataIndex::SampleSize] = 0;
        outputData[OutDataIndex::BoxEst] = 0; // Count only the ROI object
    }

    outputData[OutDataIndex::SelectedSampleSize] = selSampleSize;
    outputData[OutDataIndex::FeatureSize] = this->keypoints.size();
    outputData[OutDataIndex::SelectedFeatureSize] = selectedFeatures;
    outputData[OutDataIndex::NumClusters] = this->selectedNumClusters > 0 ? this->selectedNumClusters : hashtable_size(this->clusterMap);
    outputData[OutDataIndex::FrameNum] = frameId;
    outputData[OutDataIndex::Validity] = validity;
    outputData[OutDataIndex::MinPts] = this->minPts;
    outputData[OutDataIndex::TruthCount] = groundTruth;
}

/**
 *
 */
void CountingResults::extendLocatedObjects(Mat& frame)
{

    for(size_t i = 1; i < this->prominentLocatedObjects.size(); i++)
    {
        LocatedObject& bxs = this->prominentLocatedObjects.at(i);
        Rect2d rec = bxs.getBoundingBox().getBox();
        // Only focus on the ROIs that do not violate the boundaries
        // of the frame
        if(rec.x < 0 || rec.y < 0 || rec.x + rec.width > frame.cols|| rec.y + rec.height > frame.rows)
        {
            continue;
        }

        this->addToClusterLocatedObjects(bxs.getBoundingBox(), frame);
    }
}

/**
 * Extract prominent objects
 */
void CountingResults::extractProminentLocatedObjects()
{    
    for(map_st::iterator it = clusterLocatedObjects.begin(); it != clusterLocatedObjects.end(); ++it)
    {
        vector<LocatedObject>& tmp = it->second;

        for(size_t j = 0; j < tmp.size(); j++)
        {
            LocatedObject& tmp_o = tmp[j];
            LocatedObject::addLocatedObject(this->prominentLocatedObjects, tmp_o);
        }
    }
}

/**
 *  
 * 
 */
void CountingResults::addToLabels(int32_t label)
{
    this->labels.push_back(label);
}

/************************************************************************************
 *   PRIVATE FUNCTIONS
 ************************************************************************************/

};
