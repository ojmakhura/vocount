#include "vocount/framed.hpp"
#include "vocount/vocutils.hpp"
#include <string>

namespace vocount
{
Framed::Framed()
{
    
}

Framed::Framed(int32_t frameId, Mat& frame, Mat& descriptors, vector<KeyPoint>& keypoints, vector<int32_t> roiFeatures, Rect2d roi, int32_t groundTruth)
{
    this->roiFeatures = roiFeatures;
    this->keypoints = keypoints;
    this->descriptors = descriptors.clone();
    this->frame = frame.clone();
    this->roi = roi;
    this->frameId = frameId;
    this->groundTruth = groundTruth;

    if(roi.area())
    {
        Mat templ = frame(roi);
        int result_cols =  frame.cols;
        int result_rows = frame.rows;

        templateMatch.create( result_rows, result_cols, CV_32FC1 );
        matchTemplate(frame, templ, templateMatch, TM_SQDIFF);
        normalize(templateMatch, templateMatch, 0, 1, NORM_MINMAX, -1, Mat());
    }
}

Framed::~Framed()
{

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
Mat& Framed::getDescriptors()
{
    return this->descriptors;
}

///
/// frame
///
Mat& Framed::getFrame()
{
    return this->frame;
}

///
/// keypoints
///
vector<KeyPoint>& Framed::getKeypoints()
{
    return this->keypoints;
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
vector<int32_t>& Framed::getRoiFeatures()
{
    return this->roiFeatures;
}

///
/// results
///
map_r& Framed::getResults()
{
    return this->results;
}

///
/// combinedLocatedObjects
///
vector<LocatedObject>& Framed::getCombinedLocatedObjects()
{
    return this->combinedLocatedObjects;
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
///
/// templateMatch
///
Mat& Framed::getTemplateMatch()
{
    return this->templateMatch;
}

/**********************************************************************************************************************
 *   PRIVATE FUNCTIONS
 **********************************************************************************************************************/

/**
 *
 */
void Framed::doCluster(CountingResults& res, Mat& dataset, int32_t kSize, int32_t step, int32_t f_minPts, bool useTwo)
{
    //CountingResults* res = new CountingResults();

    int32_t m_pts = step * f_minPts;
    hdbscan scan(m_pts);

    IntIntListMap* c_map = NULL;
    IntDistancesMap* d_map = NULL;
    clustering_stats stats;
    int32_t val = -1;
    int32_t i = 0;
   
    while(val <= 2 && i < 5)
    {
        if(m_pts == (step * f_minPts))
        {
            scan.run(dataset.ptr<float>(), dataset.rows, dataset.cols, TRUE, H_FLOAT);
        }
        else
        {
            scan.reRun(m_pts);
        }

        /// Only create the cluster map for the first kSize points which
        /// belong to the current frame
        c_map = hdbscan_create_cluster_map(scan.clusterLabels, 0, kSize);
        d_map = hdbscan_get_min_max_distances(&scan, c_map);
        hdbscan_calculate_stats(d_map, &stats);
        val = hdbscan_analyse_stats(&stats);

        if(c_map != NULL)
        {
            uint hsize = res.getClusterMap() == NULL ? 0 : hashtable_size(res.getClusterMap());
            if(hashtable_size(c_map) > hsize || val > res.getValidity())
            {
                if(res.getClusterMap() != NULL)
                {
                    hdbscan_destroy_cluster_map(res.getClusterMap());
                    res.setClusterMap(NULL);
                }

                if(res.getDistancesMap() != NULL)
                {
                    hdbscan_destroy_distance_map(res.getDistancesMap());
                    res.setDistancesMap(NULL);
                }

                if(!(res.getLabels().empty()))
                {
                    res.getLabels().clear();
                }

                res.setClusterMap(c_map);
                res.setDistancesMap(d_map);
                res.getLabels().insert(res.getLabels().begin(), scan.clusterLabels, scan.clusterLabels + keypoints.size());
                res.setMinPts(m_pts);
                res.setValidity(val);
                res.setStats(stats);
            }
            else
            {
                hdbscan_destroy_cluster_map(c_map);
                hdbscan_destroy_distance_map(d_map);
            }
        }

        printf("Testing minPts = %d with validity = %d and cluster map size = %d\n", m_pts, val, hashtable_size(c_map));
        i++;
        m_pts = (f_minPts + i) * step;
    }

    //printf("Testing minPts = %d with validity = %d and cluster map size = %d\n", m_pts, val, hashtable_size(c_map));

    /// The validity less than 2 so we force oversegmentation
    if(res.getValidity() <= 2 && useTwo)
    {
        cout << "Could not detect optimum clusters. Will force over-segmentation of the clusters." << endl;
        if(res.getClusterMap() != NULL)
        {
            hdbscan_destroy_cluster_map(res.getClusterMap());
            res.setClusterMap(NULL);
        }

        if(res.getDistancesMap() != NULL)
        {
            hdbscan_destroy_distance_map(res.getDistancesMap());
            res.setDistancesMap(NULL);
        }

        if(!(res.getLabels().empty()))
        {
            res.getLabels().clear();
        }

        m_pts = step * 2;
        scan.reRun(m_pts);
        c_map = hdbscan_create_cluster_map(scan.clusterLabels, 0, kSize);
        d_map = hdbscan_get_min_max_distances(&scan, c_map);
        hdbscan_calculate_stats(d_map, &stats);
        val = hdbscan_analyse_stats(&stats);

        res.setClusterMap(c_map);
        res.setDistancesMap(d_map);
        res.getLabels().insert(res.getLabels().begin(), scan.clusterLabels, scan.clusterLabels + keypoints.size());
        res.setMinPts(m_pts);
        res.setValidity(val);
        res.setStats(stats);
    }

    printf("Selected minPts = %d and cluster table has %d\n", res.getMinPts(), hashtable_size(res.getClusterMap()));
}

void Framed::doDetectDescriptorsClusters(CountingResults& res, Mat& dataset, vector<KeyPoint>& keypoints, int32_t minPts, int32_t iterations)
{
    res.setDataset(dataset);
    res.setKeypoints(keypoints);
    res.addToClusterLocatedObjects(VRoi(this->roi), this->frame);

    /**
     * Organise points into the best possible structure. This requires
     * putting the points into the structure that has the best match to
     * the original. We use histograms to match.
     */
    res.extractProminentLocatedObjects();

    for(int32_t i = 0; i < iterations; i++)
    {
        res.extendLocatedObjects(this->frame);
        res.extractProminentLocatedObjects();
    }

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
void Framed::detectDescriptorsClusters(CountingResults& res, Mat& dataset, vector<KeyPoint>& keypoints, int32_t kSize, int32_t minPts, int32_t step, int32_t iterations, bool useTwo)
{
    doCluster(res, dataset, kSize, step, minPts, useTwo);

    // Since we forced over-segmentation of the clusters
    // we must make it up by extending the box structures
    if(res.getMinPts() == 2 && minPts > 2)
    {
        iterations += 1;
    }

    this->doDetectDescriptorsClusters(res, dataset, keypoints, minPts, iterations);
    printf("Detected %lu objects\n\n", res.getProminentLocatedObjects().size());
}

/***
 *
 */
void Framed::filterDescriptorClustersWithColourModel(CountingResults& res, vector<int32_t>& indices, int32_t minPts, int32_t iterations, VAdditions additions)
{
    CountingResults& d_res = this->getResults(ResultIndex::Descriptors);
    vector<int32_t>& d_labels = d_res.getLabels();

    label_t* labels = new label_t[indices.size()];
    // IntIntListMap* c_map = g_hash_table_new(g_int_hash, g_int_equal);
    set<int32_t> colourModelLabels;
    /// Get all clusters for the colour model.
    ///#pragma parallel
    for(size_t i = 0; i < indices.size(); i++)
    {
        int32_t label = d_labels.at(indices.at(i));
        res.getLabels().push_back(label);        
        labels[i] = label;
        colourModelLabels.insert(label);
    }

    
    IntDistancesMap* d_map = hashtable_init(45, H_INT, H_PTR, int_compare);; //g_hash_table_new_full(g_int_hash, g_int_equal, free, free); /// distance map
    IntIntListMap* c_map = hdbscan_create_cluster_map(labels, 0, indices.size());
    /// Create a new cluster and distance maps based on the labels of the colour model in the frame feature clusters
    for(set<int32_t>::iterator it = colourModelLabels.begin(); it != colourModelLabels.end(); ++it)
    {
        /// Distance map
        int32_t *d_lb = (int32_t *)malloc(sizeof(int32_t));
        *d_lb = *it;

        distance_values dl;
        
        (distance_values *)hashtable_lookup(d_res.getDistancesMap(), d_lb, &dl);
        distance_values* t_dl = (distance_values *)malloc(sizeof(distance_values));

        t_dl->min_cr = dl.min_cr;
        t_dl->max_cr = dl.max_cr;
        t_dl->cr_confidence = dl.cr_confidence;

        t_dl->min_dr = dl.min_dr;
        t_dl->max_dr = dl.max_dr;
        t_dl->dr_confidence = dl.dr_confidence;

        hashtable_insert(d_map, d_lb, t_dl);
    }

    clustering_stats stats;
    hdbscan_calculate_stats(d_map, &stats);
    int32_t val = hdbscan_analyse_stats(&stats);

    res.setStats(stats);
    res.setValidity(val);
    res.setClusterMap(c_map);
    res.setDistancesMap(d_map);
    res.setMinPts(d_res.getMinPts());

    // Since we forced over-segmentation of the clusters
    // we must make it up by extending the box structures
    if(res.getMinPts() == 2 && minPts > 2)
    {
        iterations += 1;
    }


    vector<KeyPoint> kps;
    VOCUtils::getVectorKeypoints(keypoints, indices, kps);
    Mat dset = VOCUtils::getDescriptorDataset(descriptors, keypoints, additions);
    this->doDetectDescriptorsClusters(res, dset, kps, d_res.getMinPts(), iterations);

    printf("Detected %lu objects\n\n", res.getProminentLocatedObjects().size());

    delete [] labels;
}

void Framed::generateAllClusterImages(ResultIndex idx, map<String, Mat>& selectedClustersImages)
{
    CountingResults& res = this->getResults(idx);
    COLOURS colours;
    
    IntIntListMap* cmap = res.getClusterMap();
    map_kp kpMap = res.getSelectedClustersPoints();

    set_t* keys = cmap->keys;

    for(int i = 0; i < keys->size; i++) {
        int32_t key;
        set_value_at(keys, i, &key);

        int32_t k = key;

        ArrayList l1;
        hashtable_lookup(cmap, &key, &l1);

        vector<KeyPoint>& kps = kpMap[key];
        VOCUtils::getListKeypoints(res.getKeypoints(), &l1, kps);

        Mat kimg = VOCUtils::drawKeyPoints(this->frame, kps, colours.red, -1);
        String ss = "img_keypoints-";
        string s = to_string(k);
        ss += s.c_str();
        distance_values dv;
        hashtable_lookup(res.getDistancesMap(), &k, &dv);

        ss += "-";
        ss += to_string((int)dv.cr_confidence);
        ss += "-";
        ss += to_string((int)dv.dr_confidence);
        selectedClustersImages[ss] = kimg.clone();
    }

	// g_hash_table_iter_init (&iter, res.getClusterMap());

	// while (g_hash_table_iter_next (&iter, &key, &value)){
	// 	int32_t* k = (int32_t *)key;
    //     ArrayList* l1 = (ArrayList*)value;

    //     vector<KeyPoint>& kps = res.getSelectedClustersPoints()[*k];
    //     VOCUtils::getListKeypoints(res.getKeypoints(), l1, kps);
    //     Mat kimg = VOCUtils::drawKeyPoints(this->frame, kps, colours.red, -1);
    //     String ss = "img_keypoints-";
    //     string s = to_string(*k);
    //     ss += s.c_str();
    //     distance_values dv;
    //     hashtable_lookup(res.getDistancesMap(), k, &dv);
        
    //     ss += "-";
    //     ss += to_string((int)dv.,cr_confidence);
    //     ss += "-";
    //     ss += to_string((int)dv.dr_confidence);
    //     selectedClustersImages[ss] = kimg.clone();
	// }
    res.generateOutputData(this->frameId, this->groundTruth, this->roiFeatures);
}

/**
 *
 */
void Framed::createResultsImages(ResultIndex idx, map<String, Mat>& selectedClustersImages, OutputType outputType)
{
    CountingResults& res = this->getResults(idx);
    
    if (outputType == OutputType::FINALIMAGES || outputType == OutputType::ALL)
    {
        res.generateSelectedClusterImages(this->frame, selectedClustersImages, outputType);
        res.createLocatedObjectsImages(selectedClustersImages);
    }
    
    res.generateOutputData(this->frameId, this->groundTruth, this->roiFeatures);
}

/**
 *
 */
void Framed::addResults(ResultIndex idx, CountingResults& res)
{
    results[idx] = res;
}

/**
 * Returns the results identified by idx. If the results do not exist,
 * it will return a newly created instance of CountingResults.
 */
CountingResults& Framed::getResults(ResultIndex idx)
{
    return results[idx];
}

void Framed::combineLocatedObjets(vector<KeyPoint>& selectedKeypoints)
{
    CountingResults& descriptorResults = results[ResultIndex::Descriptors];
    CountingResults& filteredResults = results[ResultIndex::SelectedKeypoints];

    //create a new vector from the ResultIndex::SelectedKeypoints structures
    // Combine the raw descriptor results and filtered descriptor results
    vector<LocatedObject> combinedObjects(descriptorResults.getProminentLocatedObjects().begin(), descriptorResults.getProminentLocatedObjects().end());

    /**
     * For each keypoint in the selected set, we find all box_structures that
     * contain the point.
     * Also a vector of vectors is not the most elegant container. A map<> would
     * be more suitable but it does not work well with OpenMP parallelisation.
     */
    vector<vector<size_t>> filteredObjects(selectedKeypoints.size(), vector<size_t>());
    #pragma omp parallel for
    for(size_t i = 0; i < filteredObjects.size(); i++)
    {
        vector<size_t>& structures = filteredObjects[i];
        KeyPoint kp = selectedKeypoints.at(i);
        for(size_t j = 0; j < combinedObjects.size(); j++)
        {
            LocatedObject& bx = combinedObjects.at(j);

            if(bx.getBoundingBox().getBox().contains(kp.pt))
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
        combinedLocatedObjects.push_back(combinedObjects.at(*it));
    }

    /// We add the colour model results located objects after filtering because
    /// we can be sure they do not contain false positives
    vector<LocatedObject>& p_objects = filteredResults.getProminentLocatedObjects();

    for(size_t i = 0; i < p_objects.size(); i++)
    {
        LocatedObject& newObject = p_objects.at(i);
        LocatedObject::addLocatedObject(combinedLocatedObjects, newObject);
    }

    printf("Colour model filtering detected %ld objects.\n\n", combinedLocatedObjects.size());
}
};
