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
			"     [-i]       						# interactive results\n"
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
					"{i||}{c||}{t||}{l||}{ta|BOOSTING|}");


	if (parser.has("help")) {
		help();
		return 0;
	}

	if(parser.has("t")){
		vcount.truth = getFrameTruth(parser.get<String>("t"));
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

		if(colourSel.minPts == -1 && (parser.has("c") || parser.has("i"))){
			printf("Finding proper value of minPts\n");
			colourSel = detectColourSelectionMinPts(frame, f.descriptors, f.keypoints);
			printf("Finding value of minPts = %d with colourSel.selectedPts as %lu from %lu\n", colourSel.minPts, colourSel.selectedDesc.rows, f.keypoints.size());
		}

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
			findROIFeature(f, colourSel);
			Mat dset = getDescriptorDataset(vcount.frameHistory, vcount.step, f.descriptors);

			results_t* res1 = do_cluster(NULL, f.descriptors, f.keypoints, vcount.step, 3, true);
			//hdbscan_print_cluster_table(res1->clusterMap);
			getSampleFeatureClusters(&f.roiFeatures, res1);
			//hdbscan_print_cluster_table(res1->roiClusterPoints);
			generateFinalPointClusters(res1->clusterMap, res1->roiClusterPoints, res1->finalPointClusters, res1->labels, res1->keypoints);
			
			boxStructure(res1->finalPointClusters, f.keypoints, f.rois[0], res1->boxStructures);
			extendBoxClusters(res1->boxStructures, f.keypoints, res1->finalPointClusters, res1->clusterMap, res1->distancesMap);
			generateClusterImages(f.frame, res1);
			createBoxStructureImages(res1->boxStructures, res1->keyPointImages);
			printf("Frame %d truth is %d\n", vcount.frameCount, vcount.truth[vcount.frameCount]);
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			res1->total = countPrint(res1->roiClusterPoints, res1->finalPointClusters, res1->cest, res1->selectedFeatures, res1->lsize);
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;

			f.results["keypoints"] = res1;
			printData(vcount, f.frame, 	f.keypoints, f.roiFeatures, *res1, f.i);
			if(parser.has("o")){
				printImages(keypointsFrameDir, res1->keyPointImages, vcount.frameCount);
			}

			if(vcount.frameHistory.size() > 0){

				// Do index based clustering
				if (parser.has("i")) {
					if(colourSel.minPts != -1){
						framed ff = vcount.frameHistory[vcount.frameHistory.size()-1];
						vector<KeyPoint> keyp(ff.keypoints.begin(), ff.keypoints.end());

						keyp.insert(keyp.end(), f.keypoints.begin(), f.keypoints.end());

						Mat dataset = getColourDataset(frame, keyp);

						hdbscan scanis(2*colourSel.minPts, DATATYPE_FLOAT);
						scanis.run(dataset.ptr<float>(), dataset.rows, dataset.cols, true);

						vector<int> f_labels(scanis.clusterLabels + ff.keypoints.size(), scanis.clusterLabels + scanis.numPoints);
						vector<int> p_labels(scanis.clusterLabels, scanis.clusterLabels + ff.keypoints.size());
						set<int> cselclusters, sss(p_labels.begin(), p_labels.end());
						map_kp clusterKeyPoints;
						//mapClusters(p_labels, clusterKeyPoints, clusterKeypointIdx, ff.keypoints);
						IntIntListMap* cmap = hdbscan_create_cluster_table(scanis.clusterLabels, 0, ff.keypoints.size());
						for(ulong i = 0; i < ff.keypoints.size(); i++){
							cselclusters.insert(scanis.clusterLabels[i]);
						}
						/****************************************************************************************************/
						printf("Selected cluster ");

						for(set<int>::iterator it = cselclusters.begin(); it != cselclusters.end(); it++){
							printf("%d ", *it);
						}
						printf("\n");
						
						GHashTableIter iter;
						gpointer key;
						gpointer value;
						g_hash_table_iter_init (&iter, cmap);

						while (g_hash_table_iter_next (&iter, &key, &value)){
							int32_t* k = (int32_t *)key;
							IntArrayList* list = (IntArrayList*)value;
							vector<KeyPoint> points;
							getListKeypoints(ff.keypoints, list, points);
							clusterKeyPoints[*k] = points;
							String s1 = to_string(*k);
							display(s1.c_str(), mm);
						}
						
						/****************************************************************************************************/
						printf(" sss.size() = %lu\n", sss.size());
						printf("f_labels has %lu and f.keypoints has %lu and cselclusters has %lu and colourSel.selectedPtsIdx has %lu\n", f_labels.size(), f.keypoints.size(), cselclusters.size(), colourSel.selectedDesc.rows);

						Mat ds;
						vector<KeyPoint> selP;
						vector<int> selPi, roiPts;
						for(size_t i = 0; i < f_labels.size(); i++){
							if(cselclusters.find(f_labels[i]) != cselclusters.end()){
								selP.push_back(f.keypoints[i]);
								selPi.push_back(i);

								if(f.roi.contains(f.keypoints[i].pt)){
									roiPts.push_back(roiPts.size());
								}

								if(ds.empty()){
									ds = f.descriptors.row(i);
								} else{
									ds.push_back(f.descriptors.row(i));
								}
							}
						}

						vector<KeyPoint> selPts;

						for (uint j = 0; j < colourSel.selectedClusters.size(); j++) {
							int cluster = colourSel.selectedClusters[j];
							vector<KeyPoint> pts = colourSel.clusterKeyPoints[cluster];
							selPts.insert(selPts.end(), pts.begin(), pts.end());
						}
						Mat mm = drawKeyPoints(frame, selPts, Scalar(0, 0, 255), -1);
						display("selP", mm);*/

						colourSel.selectedDesc = ds.clone();
						//colourSel.selectedPts = selP;
						colourSel.roiFeatures = roiPts;
						//colourSel.selectedPtsIdx = selPi;
						colourSel.clusterKeyPoints = clusterKeyPoints;
						colourSel.clusterKeypointIdx = clusterKeypointIdx;

						//printf("colourSel.selectedDesc has %d rows and colourSel.selectedPts has %lu size\n", colourSel.selectedDesc.rows, colourSel.selectedPts.size());

						vector<KeyPoint> selPts;

						for (uint j = 0; j < colourSel.selectedClusters.size();
								j++) {
							int cluster = colourSel.selectedClusters[j];
							vector<KeyPoint> pts = colourSel.clusterKeyPoints[cluster];
							selPts.insert(selPts.end(), pts.begin(), pts.end());
						}

						ds = getPointDataset(selPts);
						results_t rs = cluster(dataset, 3, true);
						rs.keypoints = selPts;
						set<int> ss(rs.labels.begin(), rs.labels.end());
						//printf("-------------- We found %lu objects by index points clustering.\n", ss.size() - 1);
						//mapClusters(rs.labels, rs.clusterKeyPoints, rs.clusterKeypointIdx, rs.keypoints);
						rs.roiClusterPoints = mapSampleFeatureClusters(colourSel.roiFeatures, rs.labels);
						generateClusterImages(f.frame, rs.clusterKeyPoints, rs.keyPointImages, rs.cest, rs.total, res1->lsize, res1->selectedFeatures);

						if(parser.has("o")){
							printf("--------------------- found %lu images\n", rs.keyPointImages.size());
							printImages(indexFrameDir, rs.keyPointImages, vcount.frameCount);
						}

						dataset = colourSel.selectedDesc.clone();
						results_t sel_r = cluster(dataset, 3, true);
						sel_r.keypoints = selPts;
						//hdbscan<float> sel_scan(_EUCLIDEAN, 3);
						//sel_scan.run(sel_r.dataset.ptr<float>(), sel_r.dataset.rows, sel_r.dataset.cols, true);
						//sel_r.labels = sel_scan.getClusterLabels();
						//mapClusters(sel_r.labels, sel_r.clusterKeyPoints, sel_r.clusterKeypointIdx, sel_r.keypoints);
						sel_r.roiClusterPoints = mapSampleFeatureClusters(colourSel.roiFeatures, sel_r.labels);
						generateClusterImages(f.frame, sel_r.clusterKeyPoints, sel_r.keyPointImages, sel_r.cest, sel_r.total, res1->lsize, res1->selectedFeatures);

						if(parser.has("o")){
							//printf("--------------------- found %lu images\n", sel_r.keyPointImages.size());
							printImages(selectedFrameDir, sel_r.keyPointImages, vcount.frameCount);
						}
						// update the colour selection so that for every new frame, colourSel is based on the previous frame.
						scanis.clean();
						//sel_scan.clean();
						//sc.clean();
					}


				}
			}*/
			
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
    	printStats(vcount.destFolder, vcount.stats);
    	printClusterEstimates(vcount.destFolder, vcount.clusterEstimates);
    }

	return 0;
}


