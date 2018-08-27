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
UMat& CountingResults::getDataset()
{
    return this->dataset;
}

void CountingResults::setDataset(UMat& dataset)
{
    dataset.copyTo(this->dataset);
    //this->dataset = dataset;
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

void CountingResults::addToClusterLocatedObjects(Rect2d roi, UMat& frame)
{
    vector<int32_t> validObjFeatures;
    VOCUtils::findValidROIFeatures(&this->keypoints, roi, &validObjFeatures, &this->labels);
    //cout << "$$$$$$$$$$$$$$$$ valid roiFeatures size " << validObjFeatures.size() << endl;

    /// There were no valid ROI features so we have to work with noise cluster
    // if(validObjFeatures.empty())
    // {
    //    VOCUtils::findROIFeatures(&this->keypoints, roi, &validObjFeatures);
    //}

    // sort the valid features by how close to the center they are
    VOCUtils::sortByDistanceFromCenter(roi, &validObjFeatures, &keypoints);

    LocatedObject mbs; /// Create a box structure based on roi
    mbs.setBox(roi);
    mbs.setBoxImage(frame(mbs.getBox()));
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

        vector<LocatedObject>& availableOjects = clusterLocatedObjects[key];
        IntArrayList* l1 = (IntArrayList*)g_hash_table_lookup(clusterMap, &key);
        KeyPoint f_point = keypoints.at(validObjFeatures[i]);
        mbs.getPoints()->insert(validObjFeatures[i]);

        /// Each point in the cluster should be inside a LocatedObject
        availableOjects.push_back(mbs);
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
                    LocatedObject::addLocatedObject(&availableOjects, &newObject);
                }
            }
        }
    }
}


void CountingResults::setClusterLocatedObjects(map_st clusterLocatedObjects)
{
    this->clusterLocatedObjects = clusterLocatedObjects;
}

///
/// selectedClustersImages
///
map<String, Mat>* CountingResults::getSelectedClustersImages()
{
    return &this->selectedClustersImages;
}

void CountingResults::setSelectedClustersImages(map<String, Mat> selectedClustersImages)
{
    this->selectedClustersImages = selectedClustersImages;
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

void CountingResults::generateSelectedClusterImages(UMat& frame)
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
        vector<LocatedObject>& rects = this->clusterLocatedObjects[key];

        for(uint i = 0; i < rects.size(); i++)
        {
            RNG rng(12345);
            Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                                  rng.uniform(0, 255));
            rectangle(kimg, rects[i].getBox(), value, 2, 8, 0);
        }
        rectangle(kimg, rects[0].getBox(), colours.red, 2, 8, 0);
        String ss = "img_keypoints-";
        string s = to_string(key);
        ss += s.c_str();
        distance_values *dv = (distance_values *)g_hash_table_lookup(this->distancesMap, &key);

        ss += "-";
        ss += to_string((int)dv->cr_confidence);
        ss += "-";
        ss += to_string((int)dv->dr_confidence);
        this->selectedClustersImages[ss] = kimg;
        kp.insert(kp.end(), kps.begin(), kps.end());
    }

    Mat mm = VOCUtils::drawKeyPoints(frame, &kp, colours.red, -1);

    String ss = "img_allkps";
    this->selectedClustersImages[ss] = mm;
}

/**
 *
 */
void CountingResults::createLocatedObjectsImages()
{
    String ss = "img_bounds";
    Mat img_bounds = this->selectedClustersImages["img_allkps"].clone();

    for (size_t i = 0; i < this->prominentLocatedObjects.size(); i++)
    {
        LocatedObject b = this->prominentLocatedObjects[i];
        RNG rng(12345);
        Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                              rng.uniform(0, 255));
        rectangle(img_bounds, b.getBox(), value, 2, 8, 0);
    }
    this->selectedClustersImages[ss] = img_bounds;
}

void CountingResults::generateOutputData(int32_t frameId, int32_t groundTruth, vector<int32_t>& roiFeatures)
{
    int selSampleSize = 0;
    if(this->clusterLocatedObjects.size() > 0)
    {
        outputData[OutDataIndex::SampleSize] = roiFeatures.size();
        outputData[OutDataIndex::BoxEst] = prominentLocatedObjects.size();

        //for(map_t::iterator it = this->roiClusterPoints.begin(); it != this->roiClusterPoints.end(); ++it)
        //{
        //selSampleSize += it->second.size();
        //}

    }
    else
    {
        outputData[OutDataIndex::SampleSize] = 0;
        outputData[OutDataIndex::BoxEst] = g_hash_table_size(clusterMap) - 1;
    }

    outputData[OutDataIndex::SelectedSampleSize] = selSampleSize;
    outputData[OutDataIndex::FeatureSize] = this->keypoints.size();
    outputData[OutDataIndex::SelectedFeatureSize] = selectedFeatures;
    outputData[OutDataIndex::NumClusters] = selectedClustersImages.size();
    outputData[OutDataIndex::FrameNum] = frameId;
    outputData[OutDataIndex::Validity] = validity;
    outputData[OutDataIndex::TruthCount] = groundTruth;
}

/**
 *
 */
void CountingResults::extendLocatedObjects(UMat& frame)
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
                if(tmp[l].getBox().contains(kp.pt))
                {
                    if(t[j] == -1)
                    {
                        t[j] = l;
                        t_cmp[j] = tmp[l].getHistogramCompare();
                        tmp[t[j]].addToPoints(data[j]);
                    }
                    else if(t_cmp[j] < tmp[l].getHistogramCompare())
                    {
                        tmp[t[j]].removeFromPoints(data[j]); // remove this point
                        t_cmp[j] = tmp[l].getHistogramCompare();
                        t[j] = l;
                        tmp[t[j]].addToPoints(data[j]);
                    }
                }
            }
        }

        for(size_t j = 0; j < tmp.size(); j++)
        {
            int idx = LocatedObject::rectExist(&prominentLocatedObjects, &tmp[j]);

            if(idx == -1)  // the structure does not exist
            {
                prominentLocatedObjects.push_back(tmp[j]);
            }
            else    /// The rect exist s merge the points
            {
                LocatedObject& strct = prominentLocatedObjects.at(idx);

                // Find out which structure is more similar to the original
                // by comparing the histograms
                if(tmp[j].getHistogramCompare() > strct.getHistogramCompare())
                {
                    strct.setBox(tmp[j].getBox());
                    strct.setMatchPoint(tmp[j].getMatchPoint());
                    strct.setBoxImage(tmp[j].getBoxImage());
                    strct.setBoxGray(tmp[j].getBoxGray());
                    strct.setHistogram(tmp[j].getHistogram());
                    strct.setHistogramCompare(tmp[j].getHistogramCompare());
                    strct.setMomentsCompare(tmp[j].getMomentsCompare());
                }
                strct.getPoints()->insert(tmp[j].getPoints()->begin(), tmp[j].getPoints()->end());

            }
        }
    }
}

/************************************************************************************
 *   PRIVATE FUNCTIONS
 ************************************************************************************/

};
