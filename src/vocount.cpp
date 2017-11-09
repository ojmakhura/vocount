//#include "hdbscan.hpp"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <opencv/cv.hpp>
#include <opencv/cxcore.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <ctime>
#include <string>
#include "vocount/process_frame.hpp"
#include "vocount/print_utils.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ximgproc::segmentation;

static void help(){
	printf( "This is a programming for estimating the number of objects in the video.\n"
	        "Usage: vocount\n"
	        "     [-v][-video]=<video>         	   	# Video file to read\n"
	        "     [-o=<output dir>]     		   	# the directly where to write to frame images\n"
			"     [-n=<sample size>]       			# the number of frames to use for sample size\n"
			"     [-w=<dataset width>]       		# the number of frames to use for dataset size\n"
			"     [-t=<truth count dir>]			# The folder that contains binary images for each frame in the video with objects tagged \n"
			"     [-s]       						# select roi from the first \n"
			"     [-k]       						# raw keypoints\n"
			"     [-i]       						# image space clustering\n"
			"     [-f]       						# filtered keypoints\n"
			"     [-c]       						# cluster analysis method \n"
	        "\n" );
}

inline cv::Ptr<cv::Tracker> createTrackerByName(cv::String name)
{
    cv::Ptr<cv::Tracker> tracker;

    if (name == "KCF")
        tracker = cv::TrackerKCF::create();
    else if (name == "TLD")
        tracker = cv::TrackerTLD::create();
    else if (name == "BOOSTING")
        tracker = cv::TrackerBoosting::create();
    else if (name == "MEDIAN_FLOW")
        tracker = cv::TrackerMedianFlow::create();
    else if (name == "MIL")
        tracker = cv::TrackerMIL::create();
    else if (name == "GOTURN")
        tracker = cv::TrackerGOTURN::create();
    else
        CV_Error(cv::Error::StsBadArg, "Invalid tracking algorithm name\n");

    return tracker;
}

int main(int argc, char** argv) {
	ocl::setUseOpenCL(true);
	Mat frame;
	VideoCapture cap;
    //BoxExtractor box;
    vocount vcount;
    selection_t colourSel;
    colourSel.minPts = -1;
	String colourDir;
	String indexDir;
	String keypointsDir;
	String selectedDir;

    Ptr<Feature2D> detector = SURF::create(500);
    MultiTracker trackers;
	vector<Ptr<Tracker>> algorithms;

	cv::CommandLineParser parser(argc, argv,
					"{help ||}{o||}{n|1|}"
					"{v||}{video||}{w|1|}{s||}"
					"{i||}{c||}{t||}{l||}{ta|BOOSTING|}"
					"{k||}{f||}");


	if (parser.has("help")) {
		help();
		return 0;
	}

	if(parser.has("t")){
		getFrameTruth(parser.get<String>("t"), vcount.truth);
	}
	
	if(parser.has("ta")){
		vcount.trackerAlgorithm = parser.get<String>("ta");
	}

	if(!processOptions(vcount, parser, cap)){
		help();
		return -1;
	}

    if( !cap.isOpened() ){
        printf("Could not open stream\n");
    	return -1;
    }

	//Create the output directories
    if(parser.has("o")){
		createDirectory(vcount.destFolder, "");

		colourDir = createDirectory(vcount.destFolder, "colour");
		indexDir = createDirectory(vcount.destFolder, "index");
		keypointsDir = createDirectory(vcount.destFolder, "keypoints");
		selectedDir = createDirectory(vcount.destFolder, "selected");
		
		if(parser.has("k")){
			String name = keypointsDir + "/estimates.csv";
			vcount.descriptorsEstimatesFile.open(name.c_str());
			vcount.descriptorsEstimatesFile << "Frame #,Sample Size,Selected Sample,Feature Size, Selected Features, # Clusters,Cluster Sum, Cluster Avg., Box Est.,Actual\n";
			
			name = keypointsDir + "/ClusterEstimates.csv";
			vcount.descriptorsClusterFile.open(name.c_str());
			vcount.descriptorsClusterFile << "Frame #,Cluster Sum, Cluster Avg., Box Est.\n";
		}
				
		if(parser.has("f")){
			String name = selectedDir + "/estimates.csv";
			vcount.selDescEstimatesFile.open(name.c_str());
			vcount.selDescEstimatesFile << "Frame #,Sample Size,Selected Sample,Feature Size, Selected Features, # Clusters,Cluster Sum, Cluster Avg., Box Est.,Actual\n";
			
			name = selectedDir + "/ClusterEstimates.csv";
			vcount.selDescClusterFile.open(name.c_str());
			vcount.selDescClusterFile << "Frame #,Cluster Sum, Cluster Avg., Box Est.\n";
		}
		
		if(parser.has("i")){
			String name = indexDir + "/estimates.csv";
			vcount.indexEstimatesFile.open(name.c_str());
			vcount.indexEstimatesFile << "Frame #,Sample Size,Selected Sample,Feature Size, Selected Features, # Clusters,Cluster Sum, Cluster Avg., Box Est.,Actual\n";
			
			name = indexDir + "/ClusterEstimates.csv";
			vcount.indexClusterFile.open(name.c_str());
			vcount.indexClusterFile << "Frame #,Cluster Sum, Cluster Avg., Box Est.\n";
		}
    }

    while(cap.read(frame))
    {
		vcount.frameCount++;
    	String colourFrameDir;
    	String indexFrameDir;
    	String keypointsFrameDir;
    	String selectedFrameDir;

        if(parser.has("o")){
    		colourFrameDir = createDirectory(colourDir, to_string(vcount.frameCount));
    		indexFrameDir = createDirectory(indexDir, to_string(vcount.frameCount));
    		keypointsFrameDir = createDirectory(keypointsDir, to_string(vcount.frameCount));
    		selectedFrameDir = createDirectory(selectedDir, to_string(vcount.frameCount));
        }

		framed f, index_f, sel_f;
		bool clusterInspect = false;

		f.i = vcount.frameCount;
		index_f.i = f.i;
		sel_f.i = f.i;

		f.frame = frame.clone();
		index_f.frame = f.frame;
		sel_f.frame = f.frame;

		cvtColor(f.frame, f.gray, COLOR_BGR2GRAY);
		detector->detectAndCompute(frame, Mat(), f.keypoints, f.descriptors);

		// Listen for a key pressed
		char c = (char) waitKey(20);
		if (c == 'q') {
			break;
		} else if (c == 's' || (parser.has("s") && !vcount.roiExtracted)) { // select a roi if c has een pressed or if the program was run with -s option
			
			Mat f2 = frame.clone();
			Rect2d boundingBox = selectROI("Select ROI", f2);
			destroyWindow("Select ROI");
			f.rois.push_back(boundingBox);
			
			for(size_t i = 0; i < f.rois.size(); i++){
				algorithms.push_back(createTrackerByName(vcount.trackerAlgorithm));
			}
			
			trackers.add(algorithms, f2, f.rois);

			vcount.roiExtracted = true;

		} else if(c == 'i'){ // inspect clusters
			clusterInspect = true;
		}

		if (vcount.roiExtracted ){
			f.rois.clear();
			trackers.update(f.frame);
			RNG rng(12345);
			Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),	rng.uniform(0, 255));
			for(size_t i = 0; i < trackers.getObjects().size(); i++){
				rectangle(frame, trackers.getObjects()[i], value, 2, 8, 0);
				f.rois.push_back(trackers.getObjects()[i]);
			}
		}

		display("frame", frame);

		if (!f.descriptors.empty()) {

			cout << "################################################################################" << endl;
			cout << "                              " << f.i << endl;
			cout << "################################################################################" << endl;
			// Create clustering dataset

			f.hasRoi = vcount.roiExtracted;
			
			if(parser.has("k")){
				f.centerFeature = findROIFeature(f.keypoints, f.descriptors, f.rois, f.roiFeatures, f.roiDesc);
				Mat dset = getDescriptorDataset(vcount.frameHistory, vcount.step, f.descriptors);

				results_t* res1 = do_cluster(NULL, f.descriptors, f.keypoints, vcount.step, 3, true);
				getSampleFeatureClusters(&f.roiFeatures, res1);
				generateFinalPointClusters(res1->clusterMap, res1->roiClusterPoints, res1->finalPointClusters, res1->labels, res1->keypoints);			
				boxStructure(res1->finalPointClusters, f.keypoints, f.rois, res1->boxStructures, frame);
				extendBoxClusters(frame, res1->boxStructures, f.keypoints, res1->finalPointClusters, res1->clusterMap, res1->distancesMap);
				generateClusterImages(f.frame, res1);
				createBoxStructureImages(res1->boxStructures, res1->keyPointImages);
				//printf("Frame %d truth is %d\n", vcount.frameCount, vcount.truth[vcount.frameCount]);
				cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
				res1->total = countPrint(res1->roiClusterPoints, res1->finalPointClusters, res1->cest, res1->selectedFeatures, res1->lsize);
				cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
				cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;

				f.results["keypoints"] = res1;
				
				if(parser.has("o")){
					generateOutputData(vcount, f.frame, f.keypoints, f.roiFeatures, res1, f.i);
					printImages(keypointsFrameDir, res1->keyPointImages, vcount.frameCount);
					printEstimates(vcount.descriptorsEstimatesFile, res1->odata);
					printClusterEstimates(vcount.descriptorsClusterFile, res1->odata, res1->cest);	
				}
			}
			if(vcount.frameHistory.size() > 0 &&(parser.has("i") || parser.has("f"))){
				
				if(colourSel.minPts == -1){
					printf("Finding proper value of minPts\n");
					colourSel = detectColourSelectionMinPts(frame, f.descriptors, f.keypoints);
					//Mat kimg = drawKeyPoints(f.frame, colourSel.selectedKeypoints, Scalar(0, 0, 255), -1);
					//display("Selected Keypoints 11", kimg);
					printf("Finding value of minPts = %d with colourSel.selectedPts as %lu from %lu\n", colourSel.minPts, colourSel.selectedDesc.rows, f.keypoints.size());
				
				} else {
					framed ff = vcount.frameHistory[vcount.frameHistory.size()-1];
					vector<KeyPoint> keyp(ff.keypoints.begin(), ff.keypoints.end());
					Mat dataset = getColourDataset(ff.frame, keyp);
					keyp.insert(keyp.end(), f.keypoints.begin(), f.keypoints.end());
					dataset.push_back(getColourDataset(frame, f.keypoints));
					dataset = dataset.clone();

					hdbscan scanis(2*colourSel.minPts, DATATYPE_FLOAT);
					scanis.run(dataset.ptr<float>(), dataset.rows, dataset.cols, true);
												
					/****************************************************************************************************/
					/** Get the hash table for the current dataset and find the mapping to clusters in prev frame
					/** and map them to selected colour map
					/****************************************************************************************************/
					IntIntListMap* prevHashTable = colourSel.clusterKeypointIdx;
					colourSel.clusterKeypointIdx = hdbscan_create_cluster_table(scanis.clusterLabels + ff.keypoints.size(), 0, f.keypoints.size());
					set<int32_t> currSelClusters, cClusters;
							
					for (set<int32_t>::iterator itt = colourSel.selectedClusters.begin(); itt != colourSel.selectedClusters.end(); ++itt) {
						int32_t cluster = *itt;
						IntArrayList* list = (IntArrayList*)g_hash_table_lookup(prevHashTable, &cluster);
						int32_t* ldata = (int32_t*)list->data;
							
						/**
						 * Since I have no idea whether the clusters from the previous frames will be clustered in the same manner
						 * I have to get the cluster with the largest number of points from selected clusters
						 **/ 
						map<int32_t, vector<int32_t>> temp;
						for(int32_t x = 0; x < list->size; x++){
							int32_t idx = ldata[x];
							int32_t newCluster = (scanis.clusterLabels)[idx];
							temp[newCluster].push_back(idx);
						}
							
						int32_t selC = -1;
						int32_t mSize = 0;
						for(map<int32_t, vector<int32_t>>::iterator it = temp.begin(); it != temp.end(); ++it){
							if(mSize < it->second.size()){
								selC = it->first;
								mSize = it->second.size();
							}
						}
						currSelClusters.insert(selC);
						printf("Prev cluster %d has been found in current frame as %d and has msize = %d\n", cluster, selC, mSize);			
					}
						
					// Need to clear the previous table map
					hdbscan_destroy_cluster_table(prevHashTable);
					
					/// Here we need to create the cluster table that maps to the f.keypoints but starting from 0 so we pass the 
					/// scanis.clusterLabels pointer that has been offset by ff.keypoints. This way we won't get a runtime error when
					/// we use the clsuters to construct the 
					
					colourSel.selectedClusters = currSelClusters;
					colourSel.selectedKeypoints.clear();
					colourSel.roiFeatures.clear();

					/****************************************************************************************************/
					/** Image space clustering
					/** -------------------------
					/** Create a dataset from the keypoints by extracting the colours and using them as the dataset
					/** hence clustering in image space
					/****************************************************************************************************/
						
					Mat selDesc;
					printf("f.keypoints = %d and f.descriptors = %d \n", f.keypoints.size(), f.descriptors.rows);
					for (set<int32_t>::iterator itt = colourSel.selectedClusters.begin(); itt != colourSel.selectedClusters.end(); ++itt) {
						
						int cluster = *itt;
						IntArrayList* list = (IntArrayList*)g_hash_table_lookup(colourSel.clusterKeypointIdx, &cluster);
						getListKeypoints(f.keypoints, list, colourSel.selectedKeypoints);
						getSelectedKeypointsDescriptors(f.descriptors, list, selDesc);
					}
						
					colourSel.selectedDesc = selDesc.clone();
					//Mat kimg = drawKeyPoints(f.frame, colourSel.selectedKeypoints, Scalar(0, 0, 255), -1);
					//display("Selected Keypoints", kimg);
					
					Mat roiDesc;
					findROIFeature(colourSel.selectedKeypoints, colourSel.selectedDesc, f.rois, colourSel.roiFeatures, roiDesc);
					
					printf("colourSel.selectedKeypoints.size = %lu selDesc.rows = %d\n", colourSel.selectedKeypoints.size(), selDesc.rows);
					
					if(parser.has("i")){
						printf("Clustering selected keypoints in image space\n");
						Mat ds = getImageSpaceDataset(colourSel.selectedKeypoints);
						results_t* idxClusterRes = do_cluster(NULL, ds, colourSel.selectedKeypoints, 1, 3, true);
						set<int> ss(idxClusterRes->labels->begin(), idxClusterRes->labels->end());
						//printf("We found %lu objects by index points clustering.\n", ss.size() - 1);
						
					}				
					
					/****************************************************************************************************/
					/** Selected Colour Model Descriptor Clustering
					/** -------------------------
					/** Create a dataset of descriptors based on the selected colour model
					/** 
					/****************************************************************************************************/
					if(parser.has("f")){
						//dataset = colourSel.selectedDesc.clone();
						printf("Clustering selected keypoints in descriptor space\n\n\n");
						results_t* selDescRes = do_cluster(NULL, colourSel.selectedDesc, colourSel.selectedKeypoints, 1, 3, true);							
						getSampleFeatureClusters(&colourSel.roiFeatures, selDescRes);
						generateFinalPointClusters(selDescRes->clusterMap, selDescRes->roiClusterPoints, 
													selDescRes->finalPointClusters, selDescRes->labels, 
													selDescRes->keypoints);
															
						boxStructure(selDescRes->finalPointClusters, colourSel.selectedKeypoints, f.rois, selDescRes->boxStructures, frame);
						extendBoxClusters(frame, selDescRes->boxStructures, colourSel.selectedKeypoints, selDescRes->finalPointClusters, 
											selDescRes->clusterMap, selDescRes->distancesMap);
						generateClusterImages(f.frame, selDescRes);
						createBoxStructureImages(selDescRes->boxStructures, selDescRes->keyPointImages);
						
						cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
						selDescRes->total = countPrint(selDescRes->roiClusterPoints, selDescRes->finalPointClusters, 
														selDescRes->cest, selDescRes->selectedFeatures, selDescRes->lsize);
						cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
													
						f.results["sel_keypoints"] = selDescRes;
						
						if(parser.has("o")){
							generateOutputData(vcount, f.frame, colourSel.selectedKeypoints, colourSel.roiFeatures, selDescRes, f.i);
							printImages(selectedFrameDir, selDescRes->keyPointImages, vcount.frameCount);
							printEstimates(vcount.selDescEstimatesFile, selDescRes->odata);
							printClusterEstimates(vcount.selDescClusterFile, selDescRes->odata, selDescRes->cest);	
						}
						
					}
				}
			}
			
			/*
			for(map<String, results_t>::iterator it = f.results.begin(); it != f.results.end(); ++it){
				printf("Cleaning results %s\n", it->first.c_str());
				cleanResult(it->second);
			}
			*/
		}

		maintaintHistory(vcount, f);
	}

    if(vcount.print){
		if(vcount.descriptorsClusterFile.is_open()){
			vcount.descriptorsClusterFile.close();
		}
				
		if(vcount.descriptorsEstimatesFile.is_open()){
			vcount.descriptorsEstimatesFile.close();
		}
		
		if(vcount.selDescClusterFile.is_open()){
			vcount.selDescClusterFile.close();
		}	
		
		if(vcount.selDescEstimatesFile.is_open()){
			vcount.selDescEstimatesFile.close();
		}
			
		if(vcount.indexClusterFile.is_open()){
			vcount.indexClusterFile.close();
		}
		
		if(vcount.indexEstimatesFile.is_open()){
			vcount.indexEstimatesFile.close();
		}
    }

	return 0;
}


