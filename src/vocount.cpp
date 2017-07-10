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
//#include "box_extractor.hpp"
#include "process_frame.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ximgproc::segmentation;
//using namespace hdbscan;


//int SAMPLE_SIZE = 1;



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

    Ptr<Feature2D> detector = SURF::create(1500);
	Ptr<Tracker> tracker;

	cv::CommandLineParser parser(argc, argv,
					"{help ||}{o||}{n|1|}"
					"{v||}{video||}{w|1|}{s||}"
					"{i||}{c||}{t||}");


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

    while(cap.read(frame))
    {
		vcount.frameCount++;
		framed f;
		bool clusterInspect = false;

		f.i = vcount.frameCount;
		f.frame = frame.clone();

		cvtColor(f.frame, f.gray, COLOR_BGR2GRAY);
		detector->detectAndCompute(frame, Mat(), f.keypoints, f.descriptors);

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

		display("frame", f.frame);

		if (!f.descriptors.empty()) {

			cout << "################################################################################" << endl;
			cout << "                              " << f.i << endl;
			cout << "################################################################################" << endl;
			// Create clustering dataset
			f.hasRoi = vcount.roiExtracted;
			findROIFeature(f);
			getDataset(vcount, f);// f.descriptors.clone();
			hdbscan scan(f.dataset, _EUCLIDEAN, vcount.step*4, vcount.step*4);
			scan.run();
			vector<Cluster*> clusters = scan.getClusters();
			// Only labels from the first n indices where n is the number of features found in f.frame
			f.labels.insert(f.labels.begin(), scan.getClusterLabels().begin(), scan.getClusterLabels().begin()+f.descriptors.rows);

			mapKeyPoints(f);
			generateFinalPointClusters(f);
			getCount(f);
			boxStructure(f);

			cout << "Cluster " << f.largest << " is the largest" << endl;
			printf("f.descriptors.rows is %d and label size is %u\n", f.descriptors.rows, scan.getClusterLabels().size());

			printf("f.keyPointImages.size() = %u\n", f.keyPointImages.size());
			printData(vcount, f);
		}

		maintaintHistory(vcount, f);
	}

    if(vcount.print){
    	printStats(vcount.destFolder, vcount.stats);
    	printClusterEstimates(vcount.destFolder, vcount.clusterEstimates);
    }

	return 0;
}


