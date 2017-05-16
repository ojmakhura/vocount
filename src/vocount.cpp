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
#include "box_extractor.hpp"
#include "process_frame.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ximgproc::segmentation;
//using namespace hdbscan;


int SAMPLE_SIZE = 1;


static void help(){
	printf( "This is a programming for estimating the number of objects in the video.\n"
	        "Usage: vocount\n"
	        "     [-v][-video]=<video>         	   	# Video file to read\n"
	        "     [--dir=<output dir>]     		   	# the directly where to write to frame images\n"
			"     [-n=<sample size>]       			# the number of frames to use for sample size\n"
			"     [-w=<dataset width>]       		# the number of frames to use for dataset size\n"
			"     [-s]       						# select roi from the first \n"
	        "\n" );
}



int main(int argc, char** argv) {
	//dummy_tester();
	ocl::setUseOpenCL(true);
	Mat frame;
	String destFolder;
	bool print = false;
	VideoCapture cap;
    BoxExtractor box;
    int frameCount = 0;
    vocount vcount;

    Ptr<Feature2D> detector = SURF::create(1500);
	Ptr<Tracker> tracker = Tracker::create("BOOSTING");

	cv::CommandLineParser parser(argc, argv, "{help ||}{dir||}{n|1|}"
			"{v||}{video||}{w|1|}{s||}");

	if(parser.has("help")){
		help();
		return 0;
	}

	if (parser.has("n")) {
		String s = parser.get<String>("n");
		SAMPLE_SIZE = atoi(s.c_str());
	}

	if (parser.has("dir")) {
		destFolder = parser.get<String>("dir");
		print = true;
		printf("Will print to %s\n", destFolder.c_str());
	}

	if(parser.has("v") || parser.has("video")){

		String video = parser.has("v") ? parser.get<String>("v") : parser.get<String>("video");
		cap.open(video);
	} else {
		printf("You did not provide the video stream to open.");
		help();
		return -1;
	}

	if(parser.has("w")){

		String s = parser.get<String>("w");
		vcount.step = atoi(s.c_str());
	} else{
		vcount.step = 1;
	}

	if (tracker == NULL) {
		cout << "***Error in the instantiation of the tracker...***\n";
		return -1;
	}

    if( !cap.isOpened() ){
        printf("Could not open stream\n");
    	return -1;
    }

    while(cap.read(frame))
    {
		++frameCount;
		framed f;
		vector<Mat> d1;
		bool clusterInspect = false;

		f.i = frameCount;
		f.frame = frame;

		cvtColor(f.frame, f.gray, COLOR_BGR2GRAY);
		detector->detectAndCompute(frame, Mat(), f.keypoints, f.descriptors);

		// Listen for a key pressed
		char c = (char) waitKey(20);
		if (c == 'q') {
			break;
		} else if (c == 's' || (parser.has("s") && !vcount.roiExtracted)) { // select a roi if c has een pressed or if the program was run with -s option
			Mat f2 = frame.clone();
			f.roi = box.extract("Select ROI", f2);

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
			rectangle(f.frame, f.roi, value, 2, 8, 0);
		}

		display("frame", frame);

		if (!f.descriptors.empty()) {
			// Create clustering dataset
			findROIFeature(vcount, f);
			uint ogsize = getDataset(vcount, f);// f.descriptors.clone();
			hdbscan scan(f.dataset, _EUCLIDEAN, vcount.step*6, vcount.step*6);
			scan.run();

			// Only labels from the first n indices where n is the number of features found in f.frame
			f.labels.insert(f.labels.begin(), scan.getClusterLabels().begin(), scan.getClusterLabels().begin()+f.descriptors.rows);

			mapKeyPoints(vcount, f, scan, ogsize);
			//drawKeyPoints(frame, f.keypoints, Scalar(0, 0, 255), -1);
			getCount(vcount, f, scan, ogsize);

			vector<KeyPoint> allPts = getAllMatchedKeypoints(f);

			Mat img_allkps = drawKeyPoints(frame, allPts, Scalar(0, 0, 255), -1);

			cout << "Cluster " << f.largest << " is the largest" << endl;
			printf("f.descriptors.rows is %d and label size is %d\n", f.descriptors.rows,
					scan.getClusterLabels().size());

			printf("f.keyPointImages.size() = %d\n", f.keyPointImages.size());

			if (print && f.roiClusterCount.size() > 0) {
				printImage(destFolder, frameCount, "frame", frame);

				Mat ff = drawKeyPoints(frame, f.keypoints, Scalar(0, 0, 255), -1);
				printImage(destFolder, frameCount, "frame_kp", ff);

				for (uint i = 0; i < f.keyPointImages.size(); ++i) {
					string s = to_string(i);
					String ss = "img_keypoints-";
					ss += s.c_str();
					printImage(destFolder, frameCount, ss, f.keyPointImages[i]);
				}

				printImage(destFolder, frameCount, "img_allkps", img_allkps);

				f.odata.push_back(f.roiFeatures.size());

				int selSampleSize = 0;

				for(map<int, int>::iterator it = f.roiClusterCount.begin(); it != f.roiClusterCount.end(); ++it){
					selSampleSize += it->second;
				}

				f.odata.push_back(selSampleSize);
				f.odata.push_back(ogsize);
				f.odata.push_back(f.selectedFeatures);
				f.odata.push_back(f.keyPointImages.size());
				f.odata.push_back(f.total);
				int32_t avg = f.total / f.keyPointImages.size();
				f.odata.push_back(avg);
				f.odata.push_back(0);
				pair<int32_t, vector<int32_t> > pp(frameCount, f.odata);
				vcount.stats.insert(pp);
				f.cest.push_back(avg);
				f.cest.push_back(f.total);
				pair<int32_t, vector<int32_t> > cpp(frameCount, f.cest);
				vcount.clusterEstimates.insert(cpp);
			}
		}

		maintaintHistory(vcount, f);

	}

    if(print){
    	printStats(destFolder, vcount.stats);
    	printClusterEstimates(destFolder, vcount.clusterEstimates);
    }

	return 0;
}


