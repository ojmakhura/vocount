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
    bool roiExtracted = false;
    int frameCount = 0;
    vocount vcount;

    Ptr<Feature2D> detector = SURF::create(1500);
	Ptr<GraphSegmentation> graphSegmenter = createGraphSegmentation();
	Ptr<Tracker> tracker = Tracker::create("BOOSTING");
    Ptr<DenseOpticalFlow> flowAlgorithm = optflow::createOptFlow_Farneback();

	cv::CommandLineParser parser(argc, argv, "{help ||}{dir||}{n|1|}"
			"{v||}{video||}{w|1|}");

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

		framed f;
		vector<Mat> d1;

		f.i = frameCount;
		display("frame", frame);
		f.frame = frame;

		cvtColor(f.frame, f.gray, COLOR_BGR2GRAY);
		runSegmentation(vcount, f, graphSegmenter, flowAlgorithm);
		if (!f.flow.empty()) {

			Mat nm;
			mergeFlowAndImage(f.flow, f.gray, nm);
			//display("nm", nm);
		}

		//printf("Rows before: %d\n", descriptors.rows);
		map<uint, vector<Point> > points;
		Mat output_image = getSegmentImage(f.segments, points);

		detector->detectAndCompute(frame, Mat(), f.keypoints, f.descriptors);
		//fdesc = desc.clone();

		if (!roiExtracted && vcount.roiDesc.size() < 1) {
			Mat f2 = frame.clone();
			vcount.roi = box.extract("Select ROI", f2);

			//initializes the tracker
			if (!tracker->init(frame, vcount.roi)) {
				cout << "***Could not initialize tracker...***\n";
				return -1;
			}

			roiExtracted = true;

		} else if (vcount.roiDesc.size() < SAMPLE_SIZE) {
			RNG rng(12345);
			tracker->update(f.frame, vcount.roi);
			vcount.samples.push_back(frame(vcount.roi).clone());
			Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),	rng.uniform(0, 255));
			rectangle(frame, vcount.roi, value, 2, 8, 0);
			vector<KeyPoint> roiPts, xp;
			Mat roiDesc;
			set<int32_t> ignore = getIgnoreSegments(vcount.roi, f.segments);

			/*for(set<int32_t>::iterator it = ignore.begin(); it != ignore.end(); ++it){
			 printf("Segment : %d\n\n", *it);
			 }*/

			// Get all keypoints inside the roi
			for (uint i = 0; i < f.keypoints.size(); ++i) {
				Point p;
				p.x = f.keypoints[i].pt.x;
				p.y = f.keypoints[i].pt.y;

				int32_t seg = f.segments.at<int32_t>(p); // get the segmentation id at point p

				// find if the segment id is listed in the ignore list
				set<int32_t>::iterator it = std::find(ignore.begin(),
						ignore.end(), seg);

				if (vcount.roi.contains(f.keypoints[i].pt)) { //&&
					//printf("Segment is %d \n\n", seg);
					if (it == ignore.end()) {
						roiPts.push_back(f.keypoints[i]);
						roiDesc.push_back(f.descriptors.row(i));
					}

				}

			}
			vcount.roiDesc.push_back(roiDesc);
			vcount.roiKeypoints.push_back(roiPts);
			//printf("found %d object keypoints\n", roiDesc.rows);

		}

		if (!f.descriptors.empty() && !vcount.roiDesc.empty()) {
			// Create clustering dataset
			uint ogsize;
			Mat dset = getDataset(vcount, f, &ogsize);// f.descriptors.clone();
			hdbscan scan(dset, _EUCLIDEAN, 4, 4);
			scan.run();

			mapKeyPoints(f, scan, ogsize);
			drawKeyPoints(frame, f.keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
			getCount(f, scan, ogsize);

			vector<KeyPoint> allPts = getAllMatchedKeypoints(f);

			Mat img_allkps = drawKeyPoints(frame, allPts, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

			cout << "Cluster " << f.largest << " is the largest" << endl;
			printf("Ogsize is %d and label size is %d\n", f.descriptors.rows,
					f.labels.size());
			double min, max;
			minMaxLoc(f.segments, &min, &max);
			printf("Max segment is %f\n", max);

			cout
					<< "--------------------------------------------------------------------------------------------"
					<< endl;
			cout
					<< "---------------------------------------Statistics-------------------------------------------"
					<< endl;
			cout
					<< "--------------------------------------------------------------------------------------------"
					<< endl;

			printf("f.keyPointImages.size() = %d\n", f.keyPointImages.size());

			if (print && f.roiClusterCount.size() > 0) {
				printImage(destFolder, frameCount, "frame", frame);

				printImage(destFolder, frameCount, "output_image",
						output_image);
				Mat ff = drawKeyPoints(frame, f.keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
				printImage(destFolder, frameCount, "frame_kp", ff);

				for (uint i = 0; i < f.keyPointImages.size(); ++i) {
					string s = to_string(i);
					String ss = "img_keypoints-";
					ss += s.c_str();
					printImage(destFolder, frameCount, ss, f.keyPointImages[i]);
				}

				printImage(destFolder, frameCount, "img_allkps", img_allkps);

				f.odata.push_back(f.descriptors.rows);
				//f.odata.push_back(f.selectedSampleSize);
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

		char c = (char) waitKey(20);
		if (c == 'q') {
			break;
		} else if (c == 's') {

		}

		maintaintHistory(vcount, f);
		++frameCount;

	}

    if(print){
    	printStats(destFolder, vcount.stats);
    	printClusterEstimates(destFolder, vcount.clusterEstimates);
    }

	return 0;
}


