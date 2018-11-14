#include "vocount/counting_results.hpp"
#include "vocount/vocutils.hpp"

namespace vocount
{

CountingResults::CountingResults()
{
    //ctor
    this->clusterMap = NULL;
    this->distancesMap = NULL;
}

CountingResults::~CountingResults()
{
    if(this->clusterMap != NULL)
    {
        hdbscan_destroy_cluster_table(this->clusterMap);
        this->clusterMap = NULL;
    }

    if(this->distancesMap != NULL)
    {
        hdbscan_destroy_distance_map_table(this->distancesMap);
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
vector<KeyPoint>* CountingResults::getKeypoints()
{
    return &this->keypoints;
}

void CountingResults::setKeypoints(vector<KeyPoint> keypoints)
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
map_kp* CountingResults::getSelectedClustersPoints()
{
    return &this->selectedClustersPoints;
}

void CountingResults::setSelectedClustersPoints(map_kp selectedClustersPoints)
{
    this->selectedClustersPoints = selectedClustersPoints;
}

///
/// Stats
///
clustering_stats* CountingResults::getStats()
{
    return &this->stats;
}

void CountingResults::setStats(clustering_stats stats)
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
map<OutDataIndex, int32_t>* CountingResults::getOutputData()
{
    return &this->outputData;
}

void CountingResults::setOutputData(map<OutDataIndex, int32_t> outputData)
{
    this->outputData = outputData;
}

///
/// labels
///
vector<int32_t>* CountingResults::getLabels()
{
    return &this->labels;
}

void CountingResults::setLabels(vector<int32_t> labels)
{
    this->labels = labels;
}

///
/// prominentLocatedObjects
///
vector<LocatedObject>* CountingResults::getProminentLocatedObjects()
{
    return &this->prominentLocatedObjects;
}

void CountingResults::setProminentLocatedObjects(vector<LocatedObject> prominentLocatedObjects)
{
    this->prominentLocatedObjects = prominentLocatedObjects;
}

///
/// clusterLocatedObjects
///
map_st* CountingResults::getClusterLocatedObjects()
{
    return &this->clusterLocatedObjects;
}

void CountingResults::addToClusterLocatedObjects(Rect2d roi, Mat& frame)
{
    vector<int32_t> validObjFeatures;
    VOCUtils::findValidROIFeatures(&this->keypoints, roi, &validObjFeatures, &this->labels);

    /// There were no valid ROI features so we have to work with noise cluster
    // if(validObjFeatures.empty())
    // {
    //    VOCUtils::findROIFeatures(&this->keypoints, roi, &validObjFeatures);
    //}

    // sort the valid features by how close to the center they are
    VOCUtils::sortByDistanceFromCenter(roi, &validObjFeatures, &keypoints);

    LocatedObject mbs; /// Create a box structure based on roi
    mbs.setBox(roi);
    mbs.setMatchTo(roi);
    Mat f_image = frame(mbs.getBox());
    mbs.setBoxImage(f_image);
    Mat h = VOCUtils::calculateHistogram(mbs.getBoxImage());
    mbs.setHistogram(h);
    mbs.setHistogramCompare(compareHist(mbs.getHistogram(), mbs.getHistogram(), CV_COMP_CORREL));
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
        IntArrayList* l1 = (IntArrayList*)g_hash_table_lookup(clusterMap, &key);
        KeyPoint f_point = keypoints.at(validObjFeatures[i]);

        /// Each point in the cluster should be inside a LocatedObject
        mbs.getPoints()->insert(validObjFeatures[i]);
        int32_t* data = (int32_t *)l1->data;

        for(int32_t j = 0; j < l1->size; j++)
        {
            KeyPoint& t_point = keypoints.at(data[j]);
            //clusterKeypoints.push_back(t_point);

            if(mbs.getBox().contains(t_point.pt))  // if the point is inside mbs, add it to mbs' points
            {
                mbs.getPoints()->insert(data[j]);
            }
            else    // else create a new mbs for it
            {

                LocatedObject newObject;
                if(LocatedObject::createNewLocatedObject(f_point, t_point, &mbs, &newObject, frame))
                {
                    newObject.getPoints()->insert(data[j]);
                    newObject.setMatchTo(mbs.getBox());
                    LocatedObject::addLocatedObject(&availableOjects, &newObject);
                }
            }
        }
        availableOjects.push_back(mbs); // TODO: some cluster match objects might not be getting updated
    }
}


void CountingResults::setClusterLocatedObjects(map_st clusterLocatedObjects)
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
void CountingResults::generateSelectedClusterImages(Mat& frame, map<String, Mat>& selectedClustersImages)
{
    COLOURS colours;
    vector<KeyPoint> kp;
    for(map_st::iterator it = clusterLocatedObjects.begin(); it != clusterLocatedObjects.end(); ++it)
    {
        int32_t key = it->first;

        IntArrayList* l1 = (IntArrayList*)g_hash_table_lookup(this->clusterMap, &key);
        vector<KeyPoint>& kps = this->selectedClustersPoints[key];
        VOCUtils::getListKeypoints(&this->keypoints, l1, &kps);

        this->selectedFeatures += kps.size();
        Mat kimg = VOCUtils::drawKeyPoints(frame, &kps, colours.red, -1);
        vector<LocatedObject>& rects = it->second;

        for(uint i = 0; i < rects.size() - 1; i++)
        {
            RNG rng(12345);
            Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                                  rng.uniform(0, 255));
            rectangle(kimg, rects.at(i).getBox(), value, 2, 8, 0);
        }

        rectangle(kimg, rects.at(rects.size() - 1).getBox(), colours.red, 2, 8, 0);

        String ss = "img_keypoints-";
        string s = to_string(key);
        ss += s.c_str();
        distance_values *dv = (distance_values *)g_hash_table_lookup(this->distancesMap, &key);

        ss += "-";
        ss += to_string((int)dv->cr_confidence);
        ss += "-";
        ss += to_string((int)dv->dr_confidence);
        selectedClustersImages[ss] = kimg.clone();
        this->selectedNumClusters += 1;
        kp.insert(kp.end(), kps.begin(), kps.end());
    }

    Mat mm = VOCUtils::drawKeyPoints(frame, &kp, colours.red, -1);

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
        rectangle(img_bounds, b.getBox(), value, 2, 8, 0);
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
        outputData[OutDataIndex::BoxEst] = g_hash_table_size(clusterMap) - 1;
    }

    outputData[OutDataIndex::SelectedSampleSize] = selSampleSize;
    outputData[OutDataIndex::FeatureSize] = this->keypoints.size();
    outputData[OutDataIndex::SelectedFeatureSize] = selectedFeatures;
    outputData[OutDataIndex::NumClusters] = this->selectedNumClusters;
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

        // Only focus on the ROIs that do not violate the boundaries
        // of the frame
        if(bxs.getBox().x < 0 || bxs.getBox().y < 0 || bxs.getBox().x + bxs.getBox().width > frame.cols|| bxs.getBox().y + bxs.getBox().height > frame.rows)
        {
            continue;
        }

        this->addToClusterLocatedObjects(bxs.getBox(), frame);
    }
}

/**
 * Extract prominent
 */
void CountingResults::extractProminentLocatedObjects()
{

    for(map_st::iterator it = clusterLocatedObjects.begin(); it != clusterLocatedObjects.end(); ++it)
    {
        vector<LocatedObject>& tmp = it->second;
        int32_t key = it->first;
        IntArrayList* l1 = (IntArrayList*)g_hash_table_lookup(clusterMap, &key);
        int32_t* data = (int32_t *)l1->data;

        int32_t t[l1->size];
        for(int32_t j = 0; j < l1->size; j++)
        {
            t[j] = -1;
        }

        double t_cmp[l1->size];

        for(int32_t j = 0; j < l1->size; j++)
        {
            KeyPoint kp = keypoints.at(data[j]);

            for(size_t l = 0; l < tmp.size(); l++)
            {
                LocatedObject& tmp_o = tmp[l];
                if(tmp_o.getBox().contains(kp.pt))
                {
                    if(t[j] == -1)
                    {
                        t[j] = l;
                        t_cmp[j] = tmp_o.getHistogramCompare();
                        tmp_o.addToPoints(data[j]);
                    }
                    else if(t_cmp[j] < tmp_o.getHistogramCompare())
                    {
                        tmp[t[j]].removeFromPoints(data[j]); // remove this point
                        t_cmp[j] = tmp_o.getHistogramCompare();
                        t[j] = l;
                        tmp[t[j]].addToPoints(data[j]);
                    }
                }
            }
        }

        for(size_t j = 0; j < tmp.size(); j++)
        {
            LocatedObject& tmp_o = tmp[j];
            int idx = LocatedObject::rectExist(&prominentLocatedObjects, &tmp_o);

            if(idx == -1)  // the structure does not exist
            {
                prominentLocatedObjects.push_back(tmp_o);
            }
            else    /// The rect exist s merge the points
            {
                LocatedObject& strct = prominentLocatedObjects.at(idx);

                // Find out which structure is more similar to the original
                // by comparing the histograms
                if(tmp_o.getHistogramCompare() > strct.getHistogramCompare())
                {
                    strct.setBox(tmp_o.getBox());
                    strct.setMatchPoint(tmp_o.getMatchPoint());
                    strct.setBoxImage(tmp_o.getBoxImage());
                    strct.setBoxGray(tmp_o.getBoxGray());
                    strct.setHistogram(tmp_o.getHistogram());
                    strct.setHistogramCompare(tmp_o.getHistogramCompare());
                    strct.setMomentsCompare(tmp_o.getMomentsCompare());
                }
                strct.getPoints()->insert(tmp_o.getPoints()->begin(), tmp_o.getPoints()->end());

            }
        }
    }
}

/************************************************************************************
 *   PRIVATE FUNCTIONS
 ************************************************************************************/

};
