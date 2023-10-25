#include <lmdb.h>
#include <chrono>

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "vocount/vocounter.hpp"
#include "vocount/voprinter.hpp"
#include "vocount/vocount_types.hpp"

using namespace chrono;

namespace vocount
{
static COLOURS colours;
VOCounter::VOCounter()
{
    this->frameCount = 0;
}

VOCounter::~VOCounter()
{

    if(descriptorsClusterFile.is_open())
    {
        descriptorsClusterFile.close();
    }

    if(descriptorsEstimatesFile.is_open())
    {
        descriptorsEstimatesFile.close();
    }

    if(cModelClusterFile.is_open())
    {
        cModelClusterFile.close();
    }

    if(cModelEstimatesFile.is_open())
    {
        cModelEstimatesFile.close();
    }

    if(combinedClusterFile.is_open())
    {
        combinedClusterFile.close();
    }

    if(combinedEstimatesFile.is_open())
    {
        combinedEstimatesFile.close();
    }

    if(filteringClusterFile.is_open())
    {
        filteringClusterFile.close();
    }

    if(filteringEstimatesFile.is_open())
    {
        filteringEstimatesFile.close();
    }

    if(trackingFile.is_open())
    {
        trackingFile.close();
    }

    for(size_t t = 0; t < framedHistory.size(); t++)
    {
        delete framedHistory[t];
        framedHistory[t] = NULL;
    }
}


/************************************************************************************
 *   PUBLIC FUNCTIONS
 ************************************************************************************/

/**
 * Read the video ground truth from the lmdb database.
 */
void VOCounter::readFrameTruth()
{

    int32_t rc;
    MDB_env *env;
    MDB_dbi dbi;
    MDB_val key, data;
    MDB_txn *txn;
    MDB_cursor *cursor;

    rc = mdb_env_create(&env);
    rc = mdb_env_open(env, settings.truthFolder.c_str(), 0, 0664);
    rc = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn);
    rc = mdb_dbi_open(txn, NULL, 0, &dbi);
    rc = mdb_cursor_open(txn, dbi, &cursor);

    while ((rc = mdb_cursor_get(cursor, &key, &data, MDB_NEXT)) == 0)
    {
        String k((char *)key.mv_data);
        k = k.substr(0, key.mv_size);

        String v((char *)data.mv_data);
        v = v.substr(0, data.mv_size);
        this->truth[stoi(k)] = stoi(v);
    }

    mdb_cursor_close(cursor);
    mdb_txn_abort(txn);
    mdb_close(env, dbi);
    mdb_env_close(env);
}

/**
 * Process the settings to read the ground truth, get the
 * roi from the
 */
void VOCounter::processSettings()
{
    /// Check if this video has truth files
    if(!settings.truthFolder.empty())
    {
        readFrameTruth();
    }

    String estimateFileHeader = "Frame #,Feature Size,Selected Features,# Clusters,Count Estimation,Ground Truth, minPts, Validity, Accuracy, Duration(secs)\n";

    /// Check if the ROI was given in the settings
    this->roi = Rect2d(settings.x, settings.y, settings.w, settings.h);

    /// If we have print setting, we create the necessary folders and files
    if(settings.print)
    {
        VOPrinter::createDirectory(settings.outputFolder, "");
        printf("Created %s directory.\n", settings.outputFolder.c_str());

        if(settings.descriptorClustering || settings.colourModelFiltering || settings.combine)
        {
            settings.descriptorDir = VOPrinter::createDirectory(settings.outputFolder, "descriptors");
            printf("Created descriptors directory at %s.\n", settings.descriptorDir.c_str());

            String name = settings.descriptorDir + "/descriptors_estimates.csv";
            this->descriptorsEstimatesFile.open(name.c_str());
            this->descriptorsEstimatesFile << estimateFileHeader;
        }

        if(settings.colourModelTracking)
        {
            settings.colourModelDir = VOPrinter::createDirectory(settings.outputFolder, "colour_model");
            printf("Created colour_model directory at %s.\n", settings.colourModelDir.c_str());

            String name = settings.colourModelDir + "/colour_model_training.csv";
            this->trainingFile.open(name.c_str());
            this->trainingFile << "minPts, Num of Clusters, Cluster 0 Size, Validity" << endl;

            name = settings.colourModelDir + "/colour_model_tracking.csv";
            this->trackingFile.open(name.c_str());
            this->trackingFile << "Frame #, Num of Points, Selected Points, MinPts, Num of Clusters, Validity, Duration(secs)\n";
        }

        if(settings.colourModelClustering || settings.combine)
        {
            settings.colourClusteringDir = VOPrinter::createDirectory(settings.colourModelDir, "clusters");
            printf("Created colour_model clustering directory at %s.\n", settings.colourClusteringDir.c_str());

            String name = settings.colourClusteringDir + "/colour_model_estimates.csv";
            this->cModelEstimatesFile.open(name.c_str());
            this->cModelEstimatesFile << estimateFileHeader;
        }

        if(settings.colourModelFiltering)
        {
            settings.filteringDir = VOPrinter::createDirectory(settings.colourModelDir, "filtering");
            printf("Created colour_model filtering directory at %s.\n", settings.filteringDir.c_str());

            String name = settings.filteringDir + "/filtering_estimates.csv";
            this->filteringEstimatesFile.open(name.c_str());
            this->filteringEstimatesFile << estimateFileHeader;
        }

        if(settings.combine)
        {
            settings.combinationDir = VOPrinter::createDirectory(settings.outputFolder, "combined");
            printf("Created combined directory at %s.\n", settings.combinationDir.c_str());

            String name = settings.combinationDir + "/combined_estimates.csv";
            this->combinedEstimatesFile.open(name.c_str());
            this->combinedEstimatesFile << "Frame #,Count Estimation,Ground Truth, Accuracy, Duration(secs)\n";
        }
    }
}

/**
 *
 */
void VOCounter::trackInitialObject(Mat& frame, Mat& descriptors, vector<KeyPoint>& keypoints, vector<int32_t>& roiFeatures)
{
    bool roiUpdate = true;

    if(!roiExtracted)
    {
        if(settings.selectROI)  // if c has been pressed or program started with -s option
        {
            roi = selectROI("Select ROI", frame);
            destroyWindow("Select ROI");
        }

        if(roi.area() > 0) 	// if there is a viable roi
        {
            tracker = VOCUtils::createTrackerByName(settings.trackerAlgorithm);
            tracker->init( frame, roi);
            roiExtracted = true;
            settings.selectROI = false;
        }

        roiUpdate = false;
    }

    if (roiExtracted && roiUpdate)
    {
        tracker->update(frame, roi);

        VOCUtils::findROIFeatures(keypoints, roi, roiFeatures);
        VOCUtils::sortByDistanceFromCenter(roi, roiFeatures, keypoints);
        Rect2d r = roi;
        double d1 = r.area();
        VOCUtils::trimRect(r, frame.rows, frame.cols, 10);
        double d2 = r.area();

        if(d2 < d1)
        {
            vector<int32_t> checkedIdxs;
            Framed* p_framed = framedHistory[framedHistory.size()-1];

            /// Filtered LocatedObjects are less likely to be off
            vector<LocatedObject>& bxs = p_framed->getCombinedLocatedObjects();

            map_r& results = p_framed->getResults();
            if(bxs.empty() && !(results.empty()) && results.find(ResultIndex::Descriptors) != results.end())
            {
                CountingResults& res = results[ResultIndex::Descriptors];
                bxs = res.getProminentLocatedObjects();
            }

            if(bxs.empty())
            {
                settings.selectROI = false;
                roiExtracted = false;
                roi = Rect2d(0, 0, 0, 0);
                return;
            }

            /**
             * select a new roi as long as either d2 < d1 or
             * no roi features were found
             */
            size_t in = 1;
            while((d2 < d1 || roiFeatures.empty()) && in < bxs.size())
            {
                roiFeatures.clear();
                Rect2d nRect = bxs.at(in).getBoundingBox().getBox();
                VOCUtils::trimRect(nRect, frame.rows, frame.cols, 10);
                in++;
                cout << nRect << endl;
                if(nRect.area() < roi.area())
                {
                    continue;
                }

                tracker = VOCUtils::createTrackerByName(settings.trackerAlgorithm);
                tracker->init(p_framed->getFrame(), nRect); /// initialise tracker on the previous frame
                tracker->update(frame, nRect); /// Update the tracker for the current frame
                r = nRect;
                d1 = r.area();
                VOCUtils::trimRect(r, frame.rows, frame.cols, 10);

                d2 = r.area();
                VOCUtils::findROIFeatures(keypoints, r, roiFeatures);
                VOCUtils::sortByDistanceFromCenter(r, roiFeatures, keypoints);
                cout << "d1 = " << d2 << " d1 = " << d1 << " ROI feature size " << roiFeatures.size() << endl;
            }

            /// We could not find a proper replacement so we pick the second object in bxs
            if(d2 < d1 || roiFeatures.empty())
            {
                Rect2d nRect = bxs.at(1).getBoundingBox().getBox();
                VOCUtils::trimRect(nRect, frame.rows, frame.cols, 10);

                tracker = VOCUtils::createTrackerByName(settings.trackerAlgorithm);
                tracker->init(p_framed->getFrame(), nRect); /// initialise tracker on the previous frame
                tracker->update(frame, nRect); /// Update the tracker for the current frame
                r = nRect;
                d1 = r.area();
                VOCUtils::trimRect(r, frame.rows, frame.cols, 10);

                d2 = r.area();
                VOCUtils::findROIFeatures(keypoints, r, roiFeatures);
                VOCUtils::sortByDistanceFromCenter(r, roiFeatures, keypoints);
            }
        }

        roi = r;
    }
}

/**
 *
 *
 */
Mat VOCounter::getDescriptorDataset(Mat& descriptors, vector<KeyPoint>& inKeypoints, vector<KeyPoint>& outKeypoints)
{
    outKeypoints.clear();
    outKeypoints.insert(outKeypoints.end(), inKeypoints.begin(), inKeypoints.end());
    Mat dset = VOCUtils::getDescriptorDataset(descriptors, inKeypoints, settings.additions);
    /// Account for multiple frames by using settings.step
    for(int32_t i = 1; i < settings.step; i++)
    {
        size_t idx = descriptorHistory.size() - i;

        if(idx >= 0)
        {
            vector<KeyPoint>& kps = keypointHistory[idx];
            outKeypoints.insert(outKeypoints.end(), kps.begin(), kps.end());

            Mat& desc = this->descriptorHistory[idx];
            Mat ds = VOCUtils::getDescriptorDataset(desc, kps, settings.additions);
            dset.push_back(dset);
        }
    }
    return dset.clone();
}

/**
 *
 *
 */
void VOCounter::processFrame(Mat& frame, Mat& descriptors, vector<KeyPoint>& keypoints)
{
    this->frameCount++;
    
    //medianBlur(frame, frame, 5);
    /// Only process is there are some keypoints
    if(!keypoints.empty())
    {
        cout << "################################################################################" << endl;
        cout << "                                        " << this->frameCount << endl;
        cout << "################################################################################" << endl;
        printf("Frame %d truth is %d\n", this->frameCount, this->truth[this->frameCount]);
        Mat fr = frame.clone();

        if(settings.colourModelTracking)
        {
            if(framedHistory.size() > 0)
            {
                cout << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Track Colour Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
                trackFrameColourModel(frame, descriptors, keypoints);
            }

            if(settings.print)
            {
                cout << "Printing colour model to " << settings.colourModelDir << endl;
                Mat frm = VOCUtils::drawKeyPoints(fr, colourModel.getSelectedKeypoints(), colours.red, -1);
                VOPrinter::printImage(settings.colourModelDir, frameCount, "frame_kp", frm);
            }
        }

        vector<int32_t> roiFeatures;
        trackInitialObject(frame, descriptors, keypoints, roiFeatures);

        RNG rng(12345);
        Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),	rng.uniform(0, 255));


        rectangle(fr, this->roi, value, 2, 8, 0);
        cout << this->roi << endl;
        VOCUtils::display("frame", fr);
        if(settings.printTracking){
            VOPrinter::printImage(settings.outputFolder, frameCount, "tracker", fr);
        }
        Framed* framed = new Framed(frameCount, frame, descriptors, keypoints, roiFeatures, this->roi, getCurrentFrameGroundTruth(this->frameCount));
        if(!roiExtracted)
        {
            this->maintainHistory(framed, descriptors, &keypoints);
            return;
        }

        /**
         * Clustering in the descriptor space with unfiltered
         * dataset.
         */
        if(settings.descriptorClustering || settings.colourModelFiltering)
        {
            cout << "\n~~~~~~~~~~~~~~~~~~~~~ Original Descriptor Space Clustering ~~~~~~~~~~~~~~~~~~~~" << endl;
            
            vector<KeyPoint> _keypoints = keypoints;
            steady_clock::time_point start = chrono::steady_clock::now();
            Mat dset = this->getDescriptorDataset(descriptors, keypoints, _keypoints);
            
            CountingResults& res = framed->getResults(ResultIndex::Descriptors);            
            framed->detectDescriptorsClusters(res, dset, _keypoints, (int32_t)keypoints.size(), settings.minPts[0], settings.step,
                                   settings.iterations[0], settings.overSegment);

            steady_clock::time_point end = chrono::steady_clock::now();
            double time = chrono::duration_cast<duration<double>>(end - start).count();

            res.setRunningTime(time);

            cout << "Original Descriptor Space Clustering completed in " << time << " seconds" << endl;

            if(settings.print)
            {
                printResults(*framed, res, ResultIndex::Descriptors, settings.descriptorDir, descriptorsEstimatesFile);
                Mat frm = VOCUtils::drawKeyPoints(fr, _keypoints, colours.red, -1);
                VOPrinter::printImage(settings.descriptorDir, frameCount, "frame_kp", frm);
            }

        }

        if(colourModel.getMinPts() >= 3 && !colourModel.getSelectedKeypoints().empty())
        {

            VOCUtils::findROIFeatures(keypoints, roi, colourModel.getRoiFeatures());

            /****************************************************************************************************/
            /// Selected Colour Model Descriptor Clustering
            /// -------------------------
            /// Create a dataset of descriptors based on the selected colour model
            ///
            /****************************************************************************************************/
            if(settings.colourModelClustering || settings.combine)
            {
                cout << "\n~~~~~~~~~~~~~~~~~~ Selected Colour Model Descriptor Clustering ~~~~~~~~~~~~~~~~" << endl;
                printf("Clustering selected keypoints in descriptor space\n\n");

                steady_clock::time_point start = chrono::steady_clock::now();
                Mat dset = VOCUtils::getDescriptorDataset(colourModel.getSelectedDesc(),
                                               colourModel.getSelectedKeypoints(),
                                               settings.additions);

                int32_t ksize = (int32_t)colourModel.getSelectedKeypoints().size();
                
                CountingResults& res = framed->getResults(ResultIndex::SelectedKeypoints);
                framed->detectDescriptorsClusters(res, dset,
                                       colourModel.getSelectedKeypoints(), ksize, settings.minPts[0], settings.step,
                                       settings.iterations[0], settings.overSegment);
                steady_clock::time_point end = chrono::steady_clock::now();
                double time = chrono::duration_cast<duration<double>>(end - start).count();

                res.setRunningTime(time);

                cout << "Selected Colour Model Descriptor Clustering completed in " << time << " seconds" << endl;

                if(settings.print)
                {
                    printResults(*framed, res, ResultIndex::SelectedKeypoints, settings.colourClusteringDir, cModelEstimatesFile);
                }
            }

            /****************************************************************************************************/
            /// Combine original descriptor clustering results with the frame colour model clustering results
            /****************************************************************************************************/
            if(settings.combine)
            {

                cout << "\n~~~~~~~~~~~~~~~ Combine descriptor and colour model locations ~~~~~~~~~~~~~~~~~" << endl;
                printf("Combinning detected objects from frame descriptors with objects from colour model\n\n");
                steady_clock::time_point start = chrono::steady_clock::now();
                framed->combineLocatedObjets(colourModel.getSelectedKeypoints());

                if(settings.print)
                {
                    Mat kimg = VOCUtils::drawKeyPoints(frame, colourModel.getSelectedKeypoints(), colours.red, -1);

                    for(size_t i = 0; i < framed->getCombinedLocatedObjects().size(); i++)
                    {
                        Scalar value;

                        RNG rng(12345);
                        value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

                        LocatedObject& b = framed->getCombinedLocatedObjects().at(i);
                        rectangle(kimg, b.getBoundingBox().getBox(), value, 2, 8, 0);
                    }
                    VOPrinter::printImage(settings.combinationDir, framed->getFrameId(), "combined", kimg) ;
                    double accuracy = 0;
                    int32_t gTruth = this->truth[framed->getFrameId()];
                    if(gTruth > 0)
                    {
                        accuracy = ((double)framed->getCombinedLocatedObjects().size() / gTruth) * 100;
                    }
                    steady_clock::time_point end = chrono::steady_clock::now();
                    double time = chrono::duration_cast<duration<double>>(end - start).count();
                    combinedEstimatesFile << framed->getFrameId() << "," <<  framed->getCombinedLocatedObjects().size() << "," << gTruth << "," << accuracy << "," << time <<"\n";
                    cout << "Combine descriptor and colour model locations completed in " << time << " seconds" << endl;
                }
            }

            /****************************************************************************************************/
            /// Filter original descriptor clustering results with the frame colour model by using the
            /// Cluster results from the original full descriptor complement
            /****************************************************************************************************/
            if(settings.colourModelFiltering)
            {
                cout << "~~~~~~~~~~~~~~~~~~~~ Selected Descriptors in Original Descriptors Clusters ~~~~~~~~~~~~~~~~~~~~~~" << endl;
                printf("Filtering by detecting colour model clusters in the full descriptor clusters\n\n");
                
                vector<int32_t>& indices = colourModel.getSelectedIndices();
                
                steady_clock::time_point start = chrono::steady_clock::now();

                CountingResults& d_res = framed->getResults(ResultIndex::DescriptorFilter);                
                framed->filterDescriptorClustersWithColourModel(d_res, indices, settings.minPts[0], settings.iterations[0], settings.additions);

                steady_clock::time_point end = chrono::steady_clock::now();
                double time = chrono::duration_cast<duration<double>>(end - start).count();
                d_res.setRunningTime(time);
                cout << "Selected Descriptors in Original Descriptors Clusters completed in " << time << " seconds" << endl;

                if(settings.print)
                {
                    printResults(*framed, d_res, ResultIndex::DescriptorFilter, settings.filteringDir, filteringEstimatesFile);
                }

            }
        }

        this->maintainHistory(framed, descriptors, &keypoints);
    }
}

/**
 *
 */
void VOCounter::trackFrameColourModel(Mat& frame, Mat& descriptors, vector<KeyPoint>& keypoints)
{
    vector<KeyPoint> keyp;
    int32_t selectedMinPts = 2 * colourModel.getMinPts();
    size_t p_size = 0;
    steady_clock::time_point start = chrono::steady_clock::now();
    Mat dataset;
    if(!framedHistory.empty())
    {
        Framed* ff = framedHistory[framedHistory.size()-1];
        keyp.insert(keyp.end(), ff->getKeypoints().begin(), ff->getKeypoints().end());
        dataset = VOCUtils::getColourDataset(ff->getFrame(), keyp);
        p_size = ff->getKeypoints().size();
    }

    keyp.insert(keyp.end(), keypoints.begin(), keypoints.end());
    dataset.push_back(VOCUtils::getColourDataset(frame, keypoints));
    dataset = dataset.clone();
    hdbscan scanis(selectedMinPts, DATATYPE_FLOAT);
    scanis.run(dataset.ptr<float>(), dataset.rows, dataset.cols, true);

    /****************************************************************************************************/
    /// Get the hash table for the current dataset and find the mapping to clusters in prev frame
    /// and map them to selected colour map
    /****************************************************************************************************/
    IntIntListMap* prevHashTable = colourModel.getColourModelClusters();
    int32_t prevNumClusters = colourModel.getNumClusters();
    IntIntListMap* t_map = hdbscan_create_cluster_map(scanis.clusterLabels + p_size, 0, keypoints.size()); // cluster map of the current frame
    colourModel.setColourModelClusters(t_map);
    colourModel.setNumClusters(g_hash_table_size(colourModel.getColourModelClusters()));
    IntDoubleListMap* distancesMap = hdbscan_get_min_max_distances(&scanis, colourModel.getColourModelClusters());
    clustering_stats stats;
    hdbscan_calculate_stats(distancesMap, &stats);
    int val = hdbscan_analyse_stats(&stats);

    if(val < 0)
    {
        cout << "Validity is less than 0. Re clustering ..." << endl;
        hdbscan_destroy_distance_map(distancesMap);
        hdbscan_destroy_cluster_map(colourModel.getColourModelClusters());

        selectedMinPts =2 * colourModel.getMinPts() - 1;
        scanis.reRun(selectedMinPts);

        prevNumClusters = colourModel.getNumClusters();
        t_map = hdbscan_create_cluster_map(scanis.clusterLabels + p_size, 0, keypoints.size());
        colourModel.setColourModelClusters(t_map);
        colourModel.setNumClusters(g_hash_table_size(colourModel.getColourModelClusters()));
        distancesMap = hdbscan_get_min_max_distances(&scanis, colourModel.getColourModelClusters());

        clustering_stats stats;
        hdbscan_calculate_stats(distancesMap, &stats);
        val = hdbscan_analyse_stats(&stats);
    }

    printf("------- MinPts = %d - new validity = %d (%d) and old validity = %d (%d)\n", selectedMinPts, val, colourModel.getNumClusters(), colourModel.getValidity(), prevNumClusters);

    colourModel.setValidity(val);
    set<int32_t> currSelClusters;

    for (set<int32_t>::iterator itt = colourModel.getSelectedClusters().begin(); itt != colourModel.getSelectedClusters().end(); ++itt)
    {
        int32_t cluster = *itt;
        ArrayList* list = (ArrayList*)hashtable_lookup(prevHashTable, &cluster);
        int32_t* ldata = (int32_t*)list->data;

        /**
         * Since we have no idea whether the clusters from the previous frames will be clustered in the same manner
         * I have to get the cluster with the largest number of points from selected clusters
         **/
        map<int32_t, vector<int32_t>> temp;
        for(int32_t x = 0; x < list->size; x++)
        {
            int32_t idx = ldata[x];
            int32_t newCluster = (scanis.clusterLabels)[idx];
            temp[newCluster].push_back(idx);
        }

        int32_t selC = -1;
        size_t mSize = 0;
        for(map<int32_t, vector<int32_t>>::iterator it = temp.begin(); it != temp.end(); ++it)
        {
            if(mSize < it->second.size())
            {
                selC = it->first;
                mSize = it->second.size();
            }

        }
        currSelClusters.insert(selC);
    }

    // Need to clear the previous table map
    hdbscan_destroy_cluster_map(prevHashTable);
    hdbscan_destroy_distance_map(distancesMap);

    colourModel.setSelectedClusters(currSelClusters);
    colourModel.getSelectedKeypoints().clear();
    colourModel.getRoiFeatures().clear();
    colourModel.getSelectedIndices().clear();

    /****************************************************************************************************/
    /// Image space clustering
    /// -------------------------
    /// Create a dataset from the keypoints by extracting the colours and using them as the dataset
    /// hence clustering in image space
    /****************************************************************************************************/

    Mat selDesc;
    for (set<int32_t>::iterator itt = colourModel.getSelectedClusters().begin(); itt != colourModel.getSelectedClusters().end(); ++itt)
    {
        int cluster = *itt;
        ArrayList* list = (ArrayList*)hashtable_lookup(colourModel.getColourModelClusters(), &cluster);
        int32_t* ldata = (int32_t*)list->data;

        colourModel.getSelectedIndices().insert(colourModel.getSelectedIndices().end(), ldata, ldata + list->size);
        vector<KeyPoint> kk;
        VOCUtils::getListKeypoints(keypoints, list, kk);
        colourModel.addToSelectedKeypoints(kk.begin(), kk.end());
        VOCUtils::getSelectedKeypointsDescriptors(descriptors, list, selDesc);

    }

    /*
    vector<KeyPoint> k2;
    VOCUtils::getVectorKeypoints(keypoints, colourModel.getSelectedIndices(), k2);
    Mat x1 = VOCUtils::drawKeyPoints(frame, k2, Scalar(255, 0, 0), -1);
    VOCUtils::display("Colour Model", x1);
    */
    steady_clock::time_point end = chrono::steady_clock::now();
    double time = duration_cast<duration<double>>(end - start).count();
    cout << "Selected " << colourModel.getSelectedKeypoints().size() << " points in " << time << " seconds."  << endl;
    if(trackingFile.is_open())
    {
        trackingFile << frameCount << "," << keypoints.size() << "," << colourModel.getSelectedKeypoints().size() << "," << selectedMinPts << "," << colourModel.getNumClusters() << "," << val << "," << time << endl;
    }

    colourModel.setSelectedDesc(selDesc);
}

/**
 *
 */
void VOCounter::getLearnedColourModel(int32_t chosen)
{
    for(map<int32_t, IntIntListMap* >::iterator it = colourModelMaps.begin(); it != colourModelMaps.end(); ++it)
    {
        if(it->first == chosen)
        {
            colourModel.setMinPts(chosen);
            colourModel.setColourModelClusters(it->second);
            colourModel.setValidity(validities[it->first - 3]);
            colourModel.setNumClusters(g_hash_table_size(colourModel.getColourModelClusters()));

            /// Remove this map from colourModelMaps
            colourModelMaps[it->first] = NULL;
        }
        else
        {
            hdbscan_destroy_cluster_map(it->second);
        }
    }
}

/**
 *
 */
void VOCounter::chooseColourModel(Mat& frame, Mat& descriptors, vector<KeyPoint>& keypoints)
{
    cout << "Use 'a' to select, 'q' to reject and 'x' to exit." << endl;
    Mat selDesc;

    GHashTableIter iter;
    gpointer key;
    gpointer value;
    g_hash_table_iter_init (&iter, colourModel.getColourModelClusters());

    while (g_hash_table_iter_next (&iter, &key, &value))
    {
        ArrayList* list = (ArrayList*)value;
        int32_t* k = (int32_t *)key;
        vector<KeyPoint> kps;
        VOCUtils::getListKeypoints(keypoints, list, kps);
        Mat m = VOCUtils::drawKeyPoints(frame, kps, colours.red, -1);
        // print the choice images
        String imName = "choice_cluster_";
        imName += std::to_string(*k).c_str();
        bool done = false;

        if(*k != 0)
        {
            String windowName = "Choose ";
            windowName += std::to_string(*k).c_str();
            windowName += "?";
            VOCUtils::display(windowName.c_str(), m);

            // Listen for a key pressed
            char c = ' ';
            while(true)
            {
                if (c == 'a')
                {
                    cout << "Chosen cluster " << *k << endl;
                    colourModel.addToSelectedClusters(*k);
                    colourModel.addToSelectedKeypoints(kps.begin(), kps.end());

                    Mat xx;
                    VOCUtils::getSelectedKeypointsDescriptors(descriptors, list, xx);
                    int32_t* ldata = (int32_t*)list->data;
                    colourModel.getSelectedIndices().insert(colourModel.getSelectedIndices().end(), ldata, ldata + list->size);

                    if(selDesc.empty())
                    {
                        selDesc = xx.clone();
                    }
                    else
                    {
                        selDesc.push_back(xx);
                    }

                    break;
                }
                else if (c == 'q')
                {
                    break;
                }
                else if (c == 'x')
                {
                    done = true;
                    break;
                }
                c = (char) waitKey(20);
            }
            destroyWindow(windowName.c_str());
        }

        if(done)
        {
            break;
        }
    }
    selDesc = selDesc.clone();
    colourModel.setSelectedDesc(selDesc);
    
}

/**
 * Detect clusters at minPts = [3, ..., 30] and find the optimum value of minPts
 * that detects the best colour model of the frame.
 *
 * @param frame - the frame to use for colour model training
 * @param keypoints - the points on the frame to use
 */
void VOCounter::trainColourModel(Mat& frame, vector<KeyPoint>& keypoints)
{

    map<uint, set<int>> numClusterMap;
    printf("Detecting minPts value for colour clustering.\n");
    steady_clock::time_point start = chrono::steady_clock::now();
    steady_clock::time_point start_three, end_three;
    hdbscan scan(3, DATATYPE_FLOAT);

    for(int i = 3; i <= 30; i++)
    {
        if(i == 3)
        {
            start_three = chrono::steady_clock::now();
            Mat dataset = VOCUtils::getColourDataset(frame, keypoints);
            scan.run(dataset.ptr<float>(), dataset.rows, dataset.cols, TRUE);
        }
        else
        {
            scan.reRun(i);
        }

        printf("\n\n >>>>>>>>>>>> Clustering for minPts = %d\n", i);
        IntIntListMap* clusterMap = hdbscan_create_cluster_map(scan.clusterLabels, 0, scan.numPoints);
        colourModelMaps[i] = clusterMap;

        IntDistancesMap* distancesMap = hdbscan_get_min_max_distances(&scan, clusterMap);
        clustering_stats stats;
        hdbscan_calculate_stats(distancesMap, &stats);
        int val = hdbscan_analyse_stats(&stats);
        uint idx = g_hash_table_size(clusterMap) - 1;
        int k = 0;
        ArrayList* p = (ArrayList *)hashtable_lookup (clusterMap, &k);

        if(p == NULL)
        {
            idx++;
        }

        printf("cluster map has size = %d and validity = %d\n", g_hash_table_size(clusterMap), val);
        if(trainingFile.is_open())
        {
            int32_t ps = 0;

            if(p != NULL)
            {
                ps = p->size;
            }

            trainingFile << i << "," << idx << "," << ps << "," << val << "\n";

            if(i == 3)
            {
                end_three = chrono::steady_clock::now();
            }
        }

        numClusterMap[idx].insert(i);
        validities.push_back(val);

        hdbscan_destroy_distance_map(distancesMap);
    }

    colourModel.setMinPts(chooseMinPts(numClusterMap, validities));
    steady_clock::time_point end = chrono::steady_clock::now();
    double time = duration_cast<duration<double>>(end - start).count();
    double time_three = duration_cast<duration<double>>(end_three - start_three).count();
    printf(">>>>>>>> OPTIMUM CHOICE OF minPts DETECTED AS %d <<<<<<<<<\n", colourModel.getMinPts());
    printf(">>>>>>>> Completed minPts = 3 in %f and final time is %f <<<<<<<<<\n", time_three, time);
    if(trainingFile.is_open())
    {
        trainingFile << "Selected minPts," << colourModel.getMinPts() << "\n";
        trainingFile << ",,,\n";
        trainingFile << ",Time at 3, Time at 4-30, Final Time, " << endl;
        trainingFile << "Training Time," << time_three << "," << time - time_three << "," << time << endl;
        trainingFile.close();
    }
}

/**
 * Get the ground truth for the current frame
 *
 * @param frameId - The current frame index
 */
int32_t VOCounter::getCurrentFrameGroundTruth(int32_t frameId)
{
    map<int32_t, int32_t>::iterator itr = this->truth.find(frameId);
    return itr == this->truth.end() ? 0 : itr->second;
}

/************************************************************************************
 *   PRIVATE FUNCTIONS
 ************************************************************************************/

/**
 *
 *
 */
int VOCounter::chooseMinPts(map<uint, set<int>>& numClusterMap, vector<int>& validities)
{
    uint numClusters = findLargestSet(numClusterMap);
    set<uint> checked;
    pair<set<uint>::iterator, bool> ret = checked.insert(numClusters);

    bool found = false;
    while(!found)
    {
        cout << "DEBUG: Largest set is " << numClusters << endl;
        set<int>& currentSelection = numClusterMap.at(numClusters);

        found = isContinuous(currentSelection);

        if(!found)
        {
            map<uint, set<int>> tmp;
            int cfirst = *(currentSelection.begin());

            for(map<uint, set<int>>::iterator it = numClusterMap.begin(); it != numClusterMap.end(); ++it)
            {
                set<int>& tmp_set = it->second;
                int tfirst = *(tmp_set.begin());
                if(it->first > numClusters && tfirst < cfirst)
                {
                    tmp[it->first] = tmp_set;
                }
            }

            numClusters = findLargestSet(tmp);
            ret = checked.insert(numClusters);

            /// If we can't insert anymore, then we are going in cycles
            /// We should use minPts = 3
            if(!ret.second)
            {
                return 3;
            }
        }
    }

    set<int> sel = numClusterMap.at(numClusters);

    return *(sel.begin());
}


/**
 * Function to assess whether the list of minPts contains a continuous
 * series of numbers ie: each value m is preceeded by a value of m-1 and
 * preceeds a value of m+1
 *
 * @param minPts - The set of minPts that result in the same number of
 * 				   clusters
 * @return where the set is continuous or not.
 */
bool VOCounter::isContinuous(set<int32_t>& minPtsList)
{
    bool continous = true;
    int32_t prev = 2;
    for(int32_t m : minPtsList)
    {
        if(prev != 2)
        {
            if(m - prev > 1)
            {
                continous = false;
                break;
            }
        }

        prev = m;
    }

    return continous;
}

/**
 * Determines whether there are more valid clustering results than
 * invalid clusters.
 *
 * @param minPtsList: the list of clusters
 * @param validities: the vector containing validities of all clustering
 * 					  results
 */
bool VOCounter::isValid(set<int32_t>& minPtsList, vector<int32_t>& validities)
{
    int32_t validCount = 0;
    int32_t invalidCount = 0;

    for(int32_t m : minPtsList)
    {
        int32_t validity = validities[m - 3]; // minPts begins at 3

        if(validity >= 0)
        {
            validCount++;
        }
        else if(validity == -2)
        {

            return false;		// If any validity is -2 then the sequence is invalid
        }
        else
        {
            invalidCount++;
        }
    }

    return validCount > invalidCount;
}

/**
 *
 */
int32_t VOCounter::findLargestSet(map<uint, set<int32_t>>& numClusterMap)
{

    uint numClusters = 0;

    /**
     * Find the first largest sequence of clustering results with the
     * same number of clusters.
     */
    for(map<uint, set<int32_t>>::iterator it = numClusterMap.begin(); it != numClusterMap.end(); ++it)
    {

        if(numClusters == 0 || it->second.size() > numClusterMap.at(numClusters).size())
        {
            numClusters = it->first;
        }
        /// If the current largest is equal to the new size, then compare
        /// the first entries
        else if(it->second.size() == numClusters)
        {

            set<int32_t>& previous = numClusterMap.at(numClusters);
            set<int32_t>& current = it->second;

            int32_t pfirst = *(previous.begin());
            int32_t cfirst = *(current.begin());

            if(cfirst > pfirst)
            {
                numClusters = it->first;
            }
        }
    }

    return numClusters;
}

/**
 * Maintain a history of 10 frame processing
 *
 * @param f - Framed object to add to the history
 */
void VOCounter::maintainHistory(Framed* framed, Mat& descriptors, vector<KeyPoint>* keypoints)
{
    if(this->framedHistory.size() == 10)
    {
        /// Framed history
        Framed *f_tmp = this->framedHistory.front();
        delete f_tmp;
        this->framedHistory.erase(this->framedHistory.begin());

        /// Keypoints history
        this->keypointHistory.erase(this->keypointHistory.begin());

        /// Descriptors history
        this->descriptorHistory.erase(this->descriptorHistory.begin());
    }
    this->framedHistory.push_back(framed);
    this->keypointHistory.push_back(*keypoints);
    this->descriptorHistory.push_back(descriptors);

}

/**
 *
 */
void VOCounter::printResults(Framed& framed, CountingResults& res, ResultIndex idx, String outDir, ofstream& estimatesFile)
{
    map<String, Mat> selectedClustersImages;
    if(settings.clustersOnly)
    {
        framed.generateAllClusterImages(idx, selectedClustersImages);
    } else {
        framed.createResultsImages(idx, selectedClustersImages, settings.outputType);
    }
    VOPrinter::printEstimates(estimatesFile, res.getOutputData(), res.getRunningTime());

    if(settings.outputType == OutputType::FINALIMAGES || settings.outputType == OutputType::ALL)
    {
        String frameDir = VOPrinter::createDirectory(outDir, to_string(frameCount));
        VOPrinter::printImages(frameDir, selectedClustersImages, frameCount);

        String minMaxFileNale = frameDir + "/min_max.csv";
        ofstream minMaxFile(minMaxFileNale.c_str());

        minMaxFile << "Cluster, Number of Points, Min CR, Max CR, CR Ratio, CR Confidence, Min DR, Max DR, DR Ratio, DR Confidence" << endl;

        IntDistancesMap* distancesMap = res.getDistancesMap();
        map_kp& selectedClustersPoints = res.getSelectedClustersPoints();

        GHashTableIter iter;
        gpointer key;
        gpointer value;
        g_hash_table_iter_init (&iter, distancesMap);

        while (g_hash_table_iter_next (&iter, &key, &value)){
            int32_t label = *((int32_t *)key);

            map_kp::iterator it = selectedClustersPoints.find(label);

            if(it != selectedClustersPoints.end())
            {
                distance_values* dv = (distance_values *)value;
                minMaxFile << label << ","  << it->second.size()<< "," << dv->min_cr << "," << dv->max_cr << ",";
                minMaxFile << (dv->min_cr / dv->max_cr) << "," << dv->cr_confidence << ",";
                minMaxFile << dv->min_dr << "," << dv->max_dr << ",";
                minMaxFile << (dv->min_dr / dv->max_dr) << "," << dv->dr_confidence<< endl;
            }
        }

        minMaxFile.flush();
        minMaxFile.close();
    }
    
}

/************************************************************************************
 *   GETTERS AND SETTERS
 ************************************************************************************/
vsettings& VOCounter::getSettings()
{
    return this->settings;
}

///
/// colourModelMaps
///
map<int32_t, IntIntListMap*>& VOCounter::getColourModelMaps()
{
    return this->colourModelMaps;
}

///
/// validities
///
vector<int32_t>& VOCounter::getValidities()
{
    return this->validities;
}

///
/// colourModel
///
ColourModel& VOCounter::getColourModel()
{
    return this->colourModel;
}

///
/// frameCount
///
int32_t VOCounter::getFrameCount()
{
    return frameCount;
}


///
/// frameCount
///
vector<Framed*>& VOCounter::getFramedHistory()
{
    return this->framedHistory;
}
};
