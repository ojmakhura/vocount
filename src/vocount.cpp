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
#include "process_frame.hpp"
#include "print_utils.hpp"

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
	Ptr<Tracker> tracker;

	cv::CommandLineParser parser(argc, argv,
					"{help ||}{o||}{n|1|}"
					"{v||}{video||}{w|1|}{s||}"
					"{i||}{c||}{t||}{l||}");


	if (parser.has("help")) {
		help();
		return 0;
	}

	if(parser.has("t")){
		vcount.truth = getFrameTruth(parser.get<String>("t"));
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
			printf("Finding value of minPts = %d\n", colourSel.minPts);
		}

		// Listen for a key pressed
		char c = (char) waitKey(20);
		if (c == 'q') {
			break;
		} else if (c == 's' || (parser.has("s") && !vcount.roiExtracted)) { // select a roi if c has een pressed or if the program was run with -s option
			tracker = TrackerBoosting::create();

			if (tracker == NULL) {
				cout << "***Error in the instantiation of the tracker...***\n";
				return -1;
			}

			Mat f2 = frame.clone();
			f.roi = selectROI("Select ROI", f2);
			destroyWindow("Select ROI");

			//initializes the tracker
			if (!tracker->init(frame, f.roi)) {
				cout << "***Could not initialize tracker...***\n";
				return -1;
			}

			vcount.roiExtracted = true;

		} else if(c == 'i'){ // inspect clusters
			clusterInspect = true;
		}

		if (vcount.roiExtracted ){
			tracker->update(f.frame, f.roi);
			RNG rng(12345);
			Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),	rng.uniform(0, 255));
			rectangle(frame, f.roi, value, 2, 8, 0);
		}

		display("frame", frame);

		if (!f.descriptors.empty()) {

			cout << "################################################################################" << endl;
			cout << "                              " << f.i << endl;
			cout << "################################################################################" << endl;
			// Create clustering dataset
			results_t res1;
			f.hasRoi = vcount.roiExtracted;
			findROIFeature(f, colourSel);
			getDataset(vcount, res1, f.descriptors);
			hdbscan<float> scan(_EUCLIDEAN, vcount.step*3);
			scan.run(res1.dataset.ptr<float>(), res1.dataset.rows, res1.dataset.cols, true);
			vector<Cluster*> clusters = scan.getClusters();
			// Only labels from the first n indices where n is the number of features found in f.frame
			res1.labels.insert(res1.labels.begin(), scan.getClusterLabels().begin(), scan.getClusterLabels().begin()+f.descriptors.rows);

			mapClusters(res1.labels, res1.clusterKeyPoints, res1.clusterKeypointIdx, f.keypoints);
			res1.roiClusterPoints = mapSampleFeatureClusters(f.roiFeatures, res1.labels);
			generateFinalPointClusters(res1.finalPointClusters, res1.roiClusterPoints, res1.clusterKeyPoints);
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			res1.total = countPrint(res1.roiClusterPoints, res1.clusterKeyPoints, res1.cest, res1.selectedFeatures, res1.lsize);
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			//cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;

			generateClusterImages(f.frame, res1.finalPointClusters, res1.keyPointImages, res1.cest, res1.total, res1.lsize, res1.selectedFeatures);
			boxStructure(res1.finalPointClusters, f.keypoints, f.roi, res1.boxStructures, res1.keyPointImages);


			//cout << "Cluster " << f.largest << " is the largest" << endl;
			printf("f.descriptors.rows is %d and label size is %lu\n", f.descriptors.rows, scan.getClusterLabels().size());

			printf("f.keyPointImages.size() = %lu\n", res1.keyPointImages.size());
			f.results[""] = res1;
			//printData(vcount, f);
			if(parser.has("o")){
				printImages(keypointsFrameDir, res1.keyPointImages, vcount.frameCount);
			}

			scan.clean();

			if(vcount.frameHistory.size() > 0){

				// Do index based clustering
				if (parser.has("i")) {
					if(colourSel.minPts != -1){
						//colourSel = detectColourSelectionMinPts(frame, f.descriptors, f.keypoints);
						framed ff = vcount.frameHistory[vcount.frameHistory.size()-1];
						Mat ds = colourSel.selectedDesc.clone();
						vector<KeyPoint> keyp(colourSel.selectedPts.begin(), colourSel.selectedPts.end());

						keyp.insert(keyp.end(), f.keypoints.begin(), f.keypoints.end());

						Mat dataset = getColourDataset(frame, keyp);

						hdbscan<float> scanis(_EUCLIDEAN, 2*colourSel.minPts);
						scanis.run(dataset.ptr<float>(), dataset.rows, dataset.cols, true);

						vector<int> f_labels(scanis.getClusterLabels().begin() + colourSel.selectedPts.size(), scanis.getClusterLabels().end());
						set<int> cselclusters(scanis.getClusterLabels().begin(), scanis.getClusterLabels().begin() + colourSel.selectedPts.size());

						//new keypoint and descriptors
						results_t rs;
						vector<int> selectedPtsIdx;
						for(size_t i = 0; i < f_labels.size(); i++){
							if(cselclusters.find(f_labels[i]) != cselclusters.end()){
								rs.keypoints.push_back(f.keypoints[i]);
								ds.push_back(f.descriptors.row(i));
								selectedPtsIdx.push_back(i);
							}
						}

						//float data[rs.keypoints.size() * 2];
						rs.dataset = getPointDataset(rs.keypoints);
						hdbscan<float> sc(_EUCLIDEAN, 3);
						sc.run(rs.dataset.ptr<float>(), rs.keypoints.size(), 2, true);
						rs.labels = sc.getClusterLabels();
						set<int> ss(rs.labels.begin(), rs.labels.end());
						printf("-------------- We found %d objects by index points clustering.\n", ss.size() - 1);
						mapClusters(rs.labels, rs.clusterKeyPoints, rs.clusterKeypointIdx, rs.keypoints);
						rs.roiClusterPoints = mapSampleFeatureClusters(f.roiFeatures, rs.labels);
						printf("----------------- rs.roiClusterPoints.size() = %d\n", rs.roiClusterPoints.size());
						//generateFinalPointClusters(rs.finalPointClusters, rs.roiClusterPoints, rs.clusterKeyPoints);
						generateClusterImages(f.frame, rs.clusterKeyPoints, rs.keyPointImages, rs.cest, rs.total, res1.lsize, res1.selectedFeatures);

						if(parser.has("o")){
							printf("--------------------- found %d images\n", rs.keyPointImages.size());
							printImages(indexFrameDir, rs.keyPointImages, vcount.frameCount);
						}

						ds = ds.clone();

						results_t sel_r;
						sel_r.dataset = colourSel.selectedDesc.clone();
						sel_r.keypoints = colourSel.selectedPts;
						hdbscan<float> sel_scan(_EUCLIDEAN, 3);
						sel_scan.run(sel_r.dataset.ptr<float>(), sel_r.dataset.rows, sel_r.dataset.cols, true);
						sel_r.labels = sel_scan.getClusterLabels(); //.insert(sel_r.labels.begin(), sel_scan.getClusterLabels().begin() + colourSel.selectedPts.size(), sel_scan.getClusterLabels().end());
						mapClusters(sel_r.labels, sel_r.clusterKeyPoints, sel_r.clusterKeypointIdx, sel_r.keypoints);
						sel_r.roiClusterPoints = mapSampleFeatureClusters(f.roiFeatures, sel_r.labels);
						//generateFinalPointClusters(sel_r.finalPointClusters, sel_r.roiClusterPoints, sel_r.clusterKeyPoints);
						generateClusterImages(f.frame, sel_r.clusterKeyPoints, sel_r.keyPointImages, sel_r.cest, sel_r.total, res1.lsize, res1.selectedFeatures);

						if(parser.has("o")){
							printf("--------------------- found %d images\n", sel_r.keyPointImages.size());
							printImages(selectedFrameDir, sel_r.keyPointImages, vcount.frameCount);
						}
						// update the colour selection so that for every new frame, colourSel is based on the previous frame.
						colourSel.selectedPtsIdx = selectedPtsIdx;
						colourSel.selectedPts = rs.keypoints;
						colourSel.selectedDesc = ds;
						scanis.clean();
						sel_scan.clean();
					}


				}
			}
		}

		maintaintHistory(vcount, f);
	}

    if(vcount.print){
    	printStats(vcount.destFolder, vcount.stats);
    	printClusterEstimates(vcount.destFolder, vcount.clusterEstimates);
    }

	return 0;
}


