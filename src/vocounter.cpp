#include "vocount/vocounter.hpp"
#include "vocount/voprinter.hpp"
#include "vocount/vocount_types.hpp"
#include <lmdb.h>

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

    if(selDescClusterFile.is_open())
    {
        selDescClusterFile.close();
    }

    if(selDescEstimatesFile.is_open())
    {
        selDescEstimatesFile.close();
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

    /// Check if the ROI was given in the settings
    this->roi = Rect2d(settings.x, settings.y, settings.w, settings.h);

    /// If we have print setting, we create the necessary folders and files
    if(settings.print)
    {
        VOPrinter::createDirectory(settings.outputFolder, "");
        printf("Created %s directory.\n", settings.outputFolder.c_str());

        if(settings.dClustering)
        {
            settings.descriptorDir = VOPrinter::createDirectory(settings.outputFolder, "descriptors");
            printf("Created descriptors directory at %s.\n", settings.descriptorDir.c_str());

            String name = settings.descriptorDir + "/estimates.csv";
            this->descriptorsEstimatesFile.open(name.c_str());
            this->descriptorsEstimatesFile << "Frame #,Feature Size,Selected Features,# Clusters,Count Estimation,Ground Truth, Validity, Accuracy\n";
        }

        if(settings.fdClustering)
        {
            settings.filteredDescDir = VOPrinter::createDirectory(settings.outputFolder, "colour_model");
            printf("Created colour_model directory at %s.\n", settings.filteredDescDir.c_str());

            String name = settings.filteredDescDir + "/estimates.csv";
            this->selDescEstimatesFile.open(name.c_str());
            this->selDescEstimatesFile << "Frame #,Feature Size, Selected Features, # Clusters, Count Estimation,Actual, Validity, Accuracy\n";

            name = settings.filteredDescDir + "/training.csv";
            this->trainingFile.open(name.c_str());
            this->trainingFile << "minPts, Num of Clusters, Cluster 0 Size, Validity" << endl;

            name = settings.filteredDescDir + "/tracking.csv";
            this->trackingFile.open(name.c_str());
            this->trackingFile << "Frame #, Num of Points, MinPts, Num of Clusters, Validity\n";

        }

        if(settings.dfClustering)
        {
            settings.dfComboDir = VOPrinter::createDirectory(settings.outputFolder, "combined");
            printf("Created combined directory at %s.\n", settings.dfComboDir.c_str());

            String name = settings.dfComboDir + "/estimates.csv";
            this->dfEstimatesFile.open(name.c_str());
            this->dfEstimatesFile << "Frame #,Count Estimation,Ground Truth, Accuracy\n";
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

        VOCUtils::findROIFeatures(&keypoints, roi, &roiFeatures);
        VOCUtils::sortByDistanceFromCenter(roi, &roiFeatures, &keypoints);
        Rect2d r = roi;
        double d1 = r.area();
        VOCUtils::trimRect(r, frame.rows, frame.cols, 10);
        double d2 = r.area();

        if(d2 < d1)
        {
            vector<int32_t> checkedIdxs;
            Framed* p_framed = framedHistory[framedHistory.size()-1];

            /// Filtered LocatedObjects are less likely to be off
            vector<LocatedObject>* bxs = p_framed->getFilteredLocatedObjects();

            if(bxs->empty() && !(p_framed->getResults()->empty()))
            {
                CountingResults* res = p_framed->getResults()->at(ResultIndex::Descriptors);
                bxs = res->getProminentLocatedObjects();
            }

            if(bxs->empty())
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
            while((d2 < d1 || roiFeatures.empty()) && in < bxs->size())
            {
                roiFeatures.clear();

                Rect2d nRect = bxs->at(in).getBox();
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
                VOCUtils::findROIFeatures(&keypoints, r, &roiFeatures);
                VOCUtils::sortByDistanceFromCenter(r, &roiFeatures, &keypoints);
                cout << "d1 = " << d2 << " d1 = " << d1 << " ROI feature size " << roiFeatures.size() << endl;
            }

            /// We could not find a proper replacement so we pick the second object in bxs
            if(d2 < d1 || roiFeatures.empty())
            {
                Rect2d nRect = bxs->at(1).getBox();
                VOCUtils::trimRect(nRect, frame.rows, frame.cols, 10);

                tracker = VOCUtils::createTrackerByName(settings.trackerAlgorithm);
                tracker->init(p_framed->getFrame(), nRect); /// initialise tracker on the previous frame
                tracker->update(frame, nRect); /// Update the tracker for the current frame
                r = nRect;
                d1 = r.area();
                VOCUtils::trimRect(r, frame.rows, frame.cols, 10);

                d2 = r.area();
                VOCUtils::findROIFeatures(&keypoints, r, &roiFeatures);
                VOCUtils::sortByDistanceFromCenter(r, &roiFeatures, &keypoints);
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
    Mat dset = VOCUtils::getDescriptorDataset(descriptors, &inKeypoints, settings.rotationalInvariance, settings.includeOctave);
    /// Account for multiple frames by using settings.step
    for(int32_t i = 1; i < settings.step; i++)
    {
        size_t idx = descriptorHistory.size() - i;

        if(idx >= 0)
        {
            vector<KeyPoint>& kps = keypointHistory[idx];
            outKeypoints.insert(outKeypoints.end(), kps.begin(), kps.end());

            Mat& desc = this->descriptorHistory[idx];
            Mat ds = VOCUtils::getDescriptorDataset(desc, &kps, settings.rotationalInvariance, settings.includeOctave);
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

    /// Only process is there are some keypoints
    if(!keypoints.empty())
    {
        cout << "################################################################################" << endl;
        cout << "                                        " << this->frameCount << endl;
        cout << "################################################################################" << endl;
        printf("Frame %d truth is %d\n", this->frameCount, this->truth[this->frameCount]);
        vector<int32_t> roiFeatures;
        trackInitialObject(frame, descriptors, keypoints, roiFeatures);

        RNG rng(12345);
        Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),	rng.uniform(0, 255));

        Mat fr = frame.clone();
        rectangle(fr, this->roi, value, 2, 8, 0);
        cout << this->roi << endl;
        VOCUtils::display("frame", fr);

        if(!roiExtracted)
        {
            return;
        }

        Framed* f = new Framed(frameCount, frame, descriptors, keypoints, roiFeatures, this->roi, getCurrentFrameGroundTruth(this->frameCount));
        //VOCUtils::display("template match", f->getTemplateMatch());

        /**
         * Clustering in the descriptor space with unfiltered
         * dataset.
         */
        if(settings.dClustering)
        {
            cout << "~~~~~~~~~~~~~~~~~~~~~ Original Descriptor Space Clustering ~~~~~~~~~~~~~~~~~~~~" << endl;

            vector<KeyPoint> _keypoints = keypoints;
            Mat dset = this->getDescriptorDataset(descriptors, keypoints, _keypoints);
            CountingResults* res = f->detectDescriptorsClusters(ResultIndex::Descriptors, dset,
                                   &_keypoints, (int32_t)keypoints.size(), settings.minPts, settings.step,
                                   settings.iterations, settings.overSegment);

            if(settings.print)
            {
                String descriptorFrameDir = VOPrinter::createDirectory(settings.descriptorDir, to_string(frameCount));
                map<String, Mat> selectedClustersImages;
                f->createResultsImages(ResultIndex::Descriptors, selectedClustersImages);
                Mat frm = VOCUtils::drawKeyPoints(fr, &_keypoints, colours.red, -1);
                VOPrinter::printImage(settings.descriptorDir, frameCount, "frame_kp", frm);
                VOPrinter::printImages(descriptorFrameDir, &selectedClustersImages, frameCount);
                VOPrinter::printEstimates(descriptorsEstimatesFile, res->getOutputData());
            }

        }

        if(colourModel.getMinPts() >= 3 && !colourModel.getSelectedKeypoints()->empty())
        {
            if(framedHistory.size() > 0)
            {
                cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Track Colour Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
                trackFrameColourModel(frame, descriptors, keypoints);
            }

            VOCUtils::findROIFeatures(&keypoints, roi, colourModel.getRoiFeatures());

            /****************************************************************************************************/
            /// Selected Colour Model Descriptor Clustering
            /// -------------------------
            /// Create a dataset of descriptors based on the selected colour model
            ///
            /****************************************************************************************************/
            if(settings.fdClustering || settings.dfClustering )
            {
                cout << "~~~~~~~~~~~~~~~~~~ Selected Colour Model Descriptor Clustering ~~~~~~~~~~~~~~~~" << endl;
                printf("Clustering selected keypoints in descriptor space\n\n");

                Mat dset = VOCUtils::getDescriptorDataset(colourModel.getSelectedDesc(),
                                               colourModel.getSelectedKeypoints(),
                                               settings.rotationalInvariance,
                                               settings.includeOctave);

                int32_t ksize = (int32_t)colourModel.getSelectedKeypoints()->size();

                CountingResults* res = f->detectDescriptorsClusters(ResultIndex::SelectedKeypoints, dset,
                                       colourModel.getSelectedKeypoints(), ksize, settings.minPts, settings.step,
                                       settings.iterations, settings.overSegment);

                if(settings.print)
                {
                    String descriptorFrameDir = VOPrinter::createDirectory(settings.filteredDescDir, to_string(frameCount));
                    map<String, Mat> selectedClustersImages;
                    f->createResultsImages(ResultIndex::SelectedKeypoints, selectedClustersImages);
                    Mat frm = VOCUtils::drawKeyPoints(fr, colourModel.getSelectedKeypoints(), colours.red, -1);
                    VOPrinter::printImage(settings.filteredDescDir, frameCount, "frame_kp", frm);
                    VOPrinter::printImages(descriptorFrameDir, &selectedClustersImages, frameCount);
                    VOPrinter::printEstimates(selDescEstimatesFile, res->getOutputData());
                }
            }

            /****************************************************************************************************/
            /// Filter original descriptor clustering results with the frame colour model
            /****************************************************************************************************/
            if(settings.dfClustering)
            {
                cout << "~~~~~~~~~~~~~~~~~~~~ Selected Descriptor Space Clustering ~~~~~~~~~~~~~~~~~~~~~~" << endl;
                printf("Filtering detected objects with colour model\n\n");
                f->filterLocatedObjets(colourModel.getSelectedKeypoints());

                if(settings.print)
                {
                    Mat kimg = VOCUtils::drawKeyPoints(frame, colourModel.getSelectedKeypoints(), colours.red, -1);

                    for(size_t i = 0; i < f->getFilteredLocatedObjects()->size(); i++)
                    {
                        Scalar value;

                        RNG rng(12345);
                        value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

                        LocatedObject& b = f->getFilteredLocatedObjects()->at(i);
                        rectangle(kimg, b.getBox(), value, 2, 8, 0);
                    }
                    VOPrinter::printImage(settings.dfComboDir, f->getFrameId(), "selected_structures", kimg) ;
                    double accuracy = 0;
                    int32_t gTruth = this->truth[f->getFrameId()];
                    if(gTruth > 0)
                    {
                        accuracy = ((double)f->getFilteredLocatedObjects()->size() / gTruth) * 100;
                    }
                    dfEstimatesFile << f->getFrameId() << "," <<  f->getFilteredLocatedObjects()->size() << "," << gTruth << "," << accuracy << "\n";
                }
            }
        }

        this->maintainHistory(f, descriptors, &keypoints);
    }
}

/**
 *
 */
void VOCounter::trackFrameColourModel(Mat& frame, Mat& descriptors, vector<KeyPoint>& keypoints)
{
    vector<KeyPoint> keyp;
    size_t p_size = 0;

    Mat dataset;
    if(!framedHistory.empty())
    {
        Framed* ff = framedHistory[framedHistory.size()-1];
        keyp.insert(keyp.end(), ff->getKeypoints()->begin(), ff->getKeypoints()->end());
        dataset = VOCUtils::getColourDataset(ff->getFrame(), &keyp);
        p_size = ff->getKeypoints()->size();
    }

    keyp.insert(keyp.end(), keypoints.begin(), keypoints.end());
    dataset.push_back(VOCUtils::getColourDataset(frame, &keypoints));
    dataset = dataset.clone();
    hdbscan scanis(2*colourModel.getMinPts(), DATATYPE_FLOAT);
    scanis.run(dataset.ptr<float>(), dataset.rows, dataset.cols, true);

    /****************************************************************************************************/
    /// Get the hash table for the current dataset and find the mapping to clusters in prev frame
    /// and map them to selected colour map
    /****************************************************************************************************/
    IntIntListMap* prevHashTable = colourModel.getColourModelClusters();
    int32_t prevNumClusters = colourModel.getNumClusters();
    IntIntListMap* t_map = hdbscan_create_cluster_table(scanis.clusterLabels + p_size, 0, keypoints.size());
    colourModel.setColourModelClusters(t_map);
    colourModel.setNumClusters(g_hash_table_size(colourModel.getColourModelClusters()));
    IntDoubleListMap* distancesMap = hdbscan_get_min_max_distances(&scanis, colourModel.getColourModelClusters());
    clustering_stats stats;
    hdbscan_calculate_stats(distancesMap, &stats);
    int val = hdbscan_analyse_stats(&stats);

    if(val < 0)
    {
        cout << "Validity is less than 0. Re clustering ..." << endl;
        hdbscan_destroy_distance_map_table(distancesMap);
        hdbscan_destroy_cluster_table(colourModel.getColourModelClusters());

        scanis.reRun(2 * colourModel.getMinPts() - 1);
        prevNumClusters = colourModel.getNumClusters();
        t_map = hdbscan_create_cluster_table(scanis.clusterLabels + p_size, 0, keypoints.size());
        colourModel.setColourModelClusters(t_map);
        colourModel.setNumClusters(g_hash_table_size(colourModel.getColourModelClusters()));
        distancesMap = hdbscan_get_min_max_distances(&scanis, colourModel.getColourModelClusters());
        clustering_stats stats;
        hdbscan_calculate_stats(distancesMap, &stats);
        val = hdbscan_analyse_stats(&stats);
    }

    printf("------- MinPts = %d - new validity = %d (%d) and old validity = %d (%d)\n", colourModel.getMinPts(), val, colourModel.getNumClusters(), colourModel.getValidity(), prevNumClusters);

    colourModel.setValidity(val);
    set<int32_t> currSelClusters;

    for (set<int32_t>::iterator itt = colourModel.getSelectedClusters()->begin(); itt != colourModel.getSelectedClusters()->end(); ++itt)
    {
        int32_t cluster = *itt;
        IntArrayList* list = (IntArrayList*)g_hash_table_lookup(prevHashTable, &cluster);
        int32_t* ldata = (int32_t*)list->data;

        /**
         * Since I have no idea whether the clusters from the previous frames will be clustered in the same manner
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
    hdbscan_destroy_cluster_table(prevHashTable);
    hdbscan_destroy_distance_map_table(distancesMap);
    colourModel.setSelectedClusters(currSelClusters);
    colourModel.getSelectedKeypoints()->clear();
    colourModel.getRoiFeatures()->clear();
    colourModel.getOldIndices()->clear();

    /****************************************************************************************************/
    /// Image space clustering
    /// -------------------------
    /// Create a dataset from the keypoints by extracting the colours and using them as the dataset
    /// hence clustering in image space
    /****************************************************************************************************/

    Mat selDesc;
    for (set<int32_t>::iterator itt = colourModel.getSelectedClusters()->begin(); itt != colourModel.getSelectedClusters()->end(); ++itt)
    {

        int cluster = *itt;
        IntArrayList* list = (IntArrayList*)g_hash_table_lookup(colourModel.getColourModelClusters(), &cluster);
        int32_t* ldata = (int32_t*)list->data;
        colourModel.getOldIndices()->insert(colourModel.getOldIndices()->end(), ldata, ldata + list->size);
        vector<KeyPoint> kk;
        VOCUtils::getListKeypoints(&keypoints, list, &kk);
        colourModel.addToSelectedKeypoints(kk.begin(), kk.end());
        VOCUtils::getSelectedKeypointsDescriptors(descriptors, list, selDesc);

    }
    cout << "Selected " << colourModel.getSelectedKeypoints()->size() << " points" << endl;

    if(trackingFile.is_open())
    {
        trackingFile << frameCount << "," << keypoints.size() << "," << colourModel.getMinPts() << "," << colourModel.getNumClusters() << "," << val << endl;
    }

    colourModel.setSelectedDesc(selDesc);
}

/**
 *
 */
void VOCounter::getLearnedColourModel(int32_t chosen)
{
    for(map<int32_t, IntDoubleListMap* >::iterator it = colourModelMaps.begin(); it != colourModelMaps.end(); ++it)
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
            hdbscan_destroy_cluster_table(it->second);
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
        IntArrayList* list = (IntArrayList*)value;
        int32_t* k = (int32_t *)key;
        vector<KeyPoint> kps;
        VOCUtils::getListKeypoints(&keypoints, list, &kps);
        Mat m = VOCUtils::drawKeyPoints(frame, &kps, colours.red, -1);
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

    hdbscan scan(3, DATATYPE_FLOAT);

    for(int i = 3; i <= 30; i++)
    {
        if(i == 3)
        {
            Mat dataset = VOCUtils::getColourDataset(frame, &keypoints);
            scan.run(dataset.ptr<float>(), dataset.rows, dataset.cols, TRUE);
        }
        else
        {
            scan.reRun(i);
        }

        printf("\n\n >>>>>>>>>>>> Clustering for minPts = %d\n", i);
        IntIntListMap* clusterMap = hdbscan_create_cluster_table(scan.clusterLabels, 0, scan.numPoints);
        colourModelMaps[i] = clusterMap;

        IntDistancesMap* distancesMap = hdbscan_get_min_max_distances(&scan, clusterMap);
        clustering_stats stats;
        hdbscan_calculate_stats(distancesMap, &stats);
        int val = hdbscan_analyse_stats(&stats);
        uint idx = g_hash_table_size(clusterMap) - 1;
        int k = 0;
        IntArrayList* p = (IntArrayList *)g_hash_table_lookup (clusterMap, &k);

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
        }

        numClusterMap[idx].insert(i);
        validities.push_back(val);

        hdbscan_destroy_distance_map_table(distancesMap);
    }

    colourModel.setMinPts(chooseMinPts(numClusterMap, validities));

    printf(">>>>>>>> OPTIMUM CHOICE OF minPts DETECTED AS %d <<<<<<<<<\n", colourModel.getMinPts());
    if(trainingFile.is_open())
    {
        trainingFile << "Selected minPts," << colourModel.getMinPts() << "\n";
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
void VOCounter::maintainHistory(Framed* f, Mat& descriptors, vector<KeyPoint>* keypoints)
{
    if(this->framedHistory.size() == 10)
    {
        /// Framed history
        Framed *f_tmp = this->framedHistory.front();
        delete f_tmp;
        this->framedHistory.erase(this->framedHistory.begin());

        /// Keypoints history
        this->keypointHistory.erase(this->keypointHistory.begin());

        /// Descritors history
        this->descriptorHistory.erase(this->descriptorHistory.begin());
    }

    this->framedHistory.push_back(f);
    this->keypointHistory.push_back(*keypoints);
    this->descriptorHistory.push_back(descriptors);

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
map<int32_t, IntIntListMap*>* VOCounter::getColourModelMaps()
{
    return &this->colourModelMaps;
}

///
/// validities
///
vector<int32_t>* VOCounter::getValidities()
{
    return &this->validities;
}

///
/// colourModel
///
ColourModel* VOCounter::getColourModel()
{
    return &this->colourModel;
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
vector<Framed*>* VOCounter::getFramedHistory()
{
    return &this->framedHistory;
}
};
