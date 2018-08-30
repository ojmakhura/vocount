#include "vocount/framed.hpp"
#include "vocount/vocutils.hpp"

namespace vocount
{
Framed::Framed()
{
    //ctor
}

Framed::Framed(int32_t frameId, UMat frame, UMat descriptors, vector<KeyPoint> keypoints, vector<int32_t> roiFeatures, Rect2d roi, int32_t groundTruth)
{
    this->roiFeatures = roiFeatures;
    this->keypoints = keypoints;
    this->descriptors = descriptors.clone();
    this->frame = frame.clone();
    this->roi = roi;
    this->frameId = frameId;
    this->groundTruth = groundTruth;
}

Framed::~Framed()
{
    for(map_r::iterator iter = results.begin(); iter != results.end(); iter++)
    {
        delete iter->second;
    }
}

/**********************************************************************************************************************
 *   GETTERS AND SETTERS
 **********************************************************************************************************************/

///
/// frameId
///
int32_t Framed::getFrameId()
{
    return this->frameId;
}

void Framed::setFrameId(int32_t frameId)
{
    this->frameId = frameId;
}


///
/// descriptors
///
UMat& Framed::getDescriptors()
{
    return this->descriptors;
}

///
/// frame
///
UMat& Framed::getFrame()
{
    return this->frame;
}

///
/// keypoints
///
vector<KeyPoint>* Framed::getKeypoints()
{
    return &this->keypoints;
}

///
/// roi
///
Rect2d Framed::getROI()
{
    return roi;
}

void Framed::setROI(Rect2d roi)
{
    this->roi = roi;
}

///
/// roiFeatures
///
vector<int32_t>* Framed::getRoiFeatures()
{
    return &this->roiFeatures;
}

///
/// results
///
map_r* Framed::getResults()
{
    return &this->results;
}

///
/// filteredLocatedObjects
///
vector<LocatedObject>* Framed::getFilteredLocatedObjects()
{
    return &this->filteredLocatedObjects;
}

///
/// groundTruth
///
int32_t Framed::getGroundTruth()
{
    return this->groundTruth;
}

void Framed::setGroundTruth(int32_t groundTruth)
{
    this->groundTruth = groundTruth;
}

/**********************************************************************************************************************
 *   PRIVATE FUNCTIONS
 **********************************************************************************************************************/

/**
 *
 */
CountingResults* Framed::doCluster(UMat& dataset, int32_t kSize, int32_t step, int32_t f_minPts, bool useTwo)
{
    CountingResults* res = new CountingResults();
    Mat mt = dataset.getMat(ACCESS_READ);

    int32_t m_pts = step * f_minPts;
    hdbscan scan(m_pts, DATATYPE_FLOAT);

    IntIntListMap* c_map = NULL;
    IntDistancesMap* d_map = NULL;
    clustering_stats stats;
    int32_t val = -1;

    int32_t i = 0;

    while(val <= 2 && i < 5)
    {
        if(m_pts == (step * f_minPts))
        {
            scan.run(mt.ptr<float>(), dataset.rows, dataset.cols, TRUE);
        }
        else
        {
            scan.reRun(m_pts);
        }

        /// Only create the cluster map for the first kSize points which
        /// belong to the current frame
        c_map = hdbscan_create_cluster_table(scan.clusterLabels, 0, kSize);

        d_map = hdbscan_get_min_max_distances(&scan, c_map);
        hdbscan_calculate_stats(d_map, &stats);
        val = hdbscan_analyse_stats(&stats);

        if(c_map != NULL)
        {
            uint hsize = res->getClusterMap() == NULL ? 0 : g_hash_table_size(res->getClusterMap());
            if(g_hash_table_size(c_map) > hsize || val > res->getValidity())
            {
                if(res->getClusterMap() != NULL)
                {
                    hdbscan_destroy_cluster_table(res->getClusterMap());
                    res->setClusterMap(NULL);
                }

                if(res->getDistancesMap() != NULL)
                {
                    hdbscan_destroy_distance_map_table(res->getDistancesMap());
                    res->setDistancesMap(NULL);
                }

                if(!(res->getLabels()->empty()))
                {
                    res->getLabels()->clear();
                }

                res->setClusterMap(c_map);
                res->setDistancesMap(d_map);
                res->getLabels()->insert(res->getLabels()->begin(), scan.clusterLabels, scan.clusterLabels + keypoints.size());
                res->setMinPts(m_pts);
                res->setValidity(val);
                res->setStats(stats);
            }
            else
            {
                hdbscan_destroy_cluster_table(c_map);
                hdbscan_destroy_distance_map_table(d_map);
            }
        }

        printf("Testing minPts = %d with validity = %d and cluster map size = %d\n", m_pts, val, g_hash_table_size(c_map));
        i++;
        m_pts = (f_minPts + i) * step;
    }

    /// The validity less than 2 so we force oversegmentation
    if(res->getValidity() <= 2 && useTwo)
    {
        cout << "Could not detect optimum clusters. Will force over-segmentation of the clusters." << endl;
        if(res->getClusterMap() != NULL)
        {
            hdbscan_destroy_cluster_table(res->getClusterMap());
            res->setClusterMap(NULL);
        }

        if(res->getDistancesMap() != NULL)
        {
            hdbscan_destroy_distance_map_table(res->getDistancesMap());
            res->setDistancesMap(NULL);
        }

        if(!(res->getLabels()->empty()))
        {
            res->getLabels()->clear();
        }

        m_pts = step * 2;
        scan.reRun(m_pts);
        c_map = hdbscan_create_cluster_table(scan.clusterLabels, 0, kSize);

        d_map = hdbscan_get_min_max_distances(&scan, c_map);
        hdbscan_calculate_stats(d_map, &stats);
        val = hdbscan_analyse_stats(&stats);

        res->setClusterMap(c_map);
        res->setDistancesMap(d_map);
        res->getLabels()->insert(res->getLabels()->begin(), scan.clusterLabels, scan.clusterLabels + keypoints.size());
        res->setMinPts(m_pts);
        res->setValidity(val);
        res->setStats(stats);
    }

    printf("Selected minPts = %d and cluster table has %d\n", res->getMinPts(), g_hash_table_size(res->getClusterMap()));

    return res;
}


/**********************************************************************************************************************
 *   PUBLIC FUNCTIONS
 **********************************************************************************************************************/

/**
 * Takes a dataset and the associated keypoints and extracts clusters. The
 * clusters are used to extract the box_structures for each clusters.
 * Prominent structures are extracted by comapring the structure in each
 * cluster.
 */
CountingResults* Framed::detectDescriptorsClusters(ResultIndex idx, UMat& dataset, vector<KeyPoint>* keypoints, int32_t kSize, int32_t step, int32_t iterations, bool useTwo)
{
    CountingResults* res = doCluster(dataset, step, 3, kSize, useTwo);
    res->setDataset(dataset);
    res->setKeypoints(*keypoints);
    res->addToClusterLocatedObjects(this->roi, this->frame);

    /**
     * Organise points into the best possible structure. This requires
     * putting the points into the structure that has the best match to
     * the original. We use histograms to match.
     */
    res->extractProminentLocatedObjects();

    // Since we forced over-segmentation of the clusters
    // we must make it up by extending the box structures
    if(res->getMinPts() == 2)
    {
        iterations += 1;
    }

    for(int32_t i = 0; i < iterations; i++)
    {
            res->extendLocatedObjects(this->frame);
            res->extractProminentLocatedObjects();
    }

    printf("boxStructure found %lu objects\n\n", res->getProminentLocatedObjects()->size());
    this->results[idx] = res;

    return res;
}


/**
 *
 */
void Framed::createResultsImages(ResultIndex idx)
{
    CountingResults* res = this->getResults(idx);
    res->generateSelectedClusterImages(this->frame);
    res->createLocatedObjectsImages();
    res->generateOutputData(this->frameId, this->groundTruth, this->roiFeatures);
}

/**
 *
 */
void Framed::addResults(ResultIndex idx, CountingResults* res)
{
    results[idx] = res;
}

/**
 *
 */
CountingResults* Framed::getResults(ResultIndex idx)
{
    return results[idx];
}

void Framed::filterLocatedObjets(vector<KeyPoint>* selectedKeypoints)
{
    CountingResults* descriptorResults = results[ResultIndex::Descriptors];
    CountingResults* filteredResults = results[ResultIndex::SelectedKeypoints];

    //create a new vector from the ResultIndex::SelectedKeypoints structures
    // Combine the raw descriptor results and filtered descriptor results
    vector<LocatedObject> combinedObjects(descriptorResults->getProminentLocatedObjects()->begin(), descriptorResults->getProminentLocatedObjects()->end());

    /**
     * For each keypoint in the selected set, we find all box_structures that
     * contain the point.
     * Also a vector of vectors is not the most elegant container. A map<> would
     * be more suitable but it does not work well with OpenMP parallelisation.
     */
    vector<vector<size_t>> filteredObjects(selectedKeypoints->size(), vector<size_t>());
    #pragma omp parallel for
    for(size_t i = 0; i < filteredObjects.size(); i++)
    {
        vector<size_t>& structures = filteredObjects[i];
        KeyPoint kp = selectedKeypoints->at(i);
        for(size_t j = 0; j < combinedObjects.size(); j++)
        {
            LocatedObject& bx = combinedObjects.at(j);

            if(bx.getBox().contains(kp.pt))
            {
                #pragma omp critical
                structures.push_back(j);
            }
        }
    }



    /**
     * For those selectedKeypoints that are inside multiple structures,
     * we find out which structure has the smallest moment comparison
     */
    set<size_t> selectedObjects;
    #pragma omp parallel for
    for(size_t i = 0; i < filteredObjects.size(); i++)
    {
        vector<size_t>& strs = filteredObjects[i];

        if(strs.size() > 0)
        {
            size_t minIdx = strs[0];
            double minMoment = combinedObjects.at(minIdx).getMomentsCompare();
            for(size_t j = 1; j < strs.size(); j++)
            {
                size_t idx = strs[j];
                double moment = combinedObjects.at(idx).getMomentsCompare();

                if(moment < minMoment)
                {
                    minIdx = idx;
                    minMoment = moment;
                }
            }
            #pragma omp critical
            selectedObjects.insert(minIdx);
        }
    }

    for(set<size_t>::iterator it = selectedObjects.begin(); it != selectedObjects.end(); it++)
    {
        filteredLocatedObjects.push_back(combinedObjects.at(*it));
    }

    /// We add the filtered results located objects after filtering because
    /// we can be sure they do not contain false positives
    vector<LocatedObject>* p_objects = filteredResults->getProminentLocatedObjects();

    for(size_t i = 0; i < p_objects->size(); i++)
    {
        LocatedObject& newObject = p_objects->at(i);
        LocatedObject::addLocatedObject(&filteredLocatedObjects, &newObject);
    }

    printf("filteredLocatedObjects.size() = %ld\n", filteredLocatedObjects.size());
}
};
