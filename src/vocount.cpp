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
#include <fstream>
#include <string>
#include "box_extractor.hpp"
#include "process_frame.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ximgproc::segmentation;
//using namespace hdbscan;


int SAMPLE_SIZE = 1;


void getSignificantSegments(vector<KeyPoint> kp, set<int> clusters){



}

static void help(){
	printf( "This is a programming for estimating the number of objects in the video.\n"
	        "Usage: vocount\n"
	        "     -[v][-video]=<video>         	   # Video file to read\n"
	        "     [--dir=<output dir>]     # the directly where to write to frame images\n"
			"     [-n=<sample size>]       # the number of frames to use for sample size"
	        "\n" );
}

void mergeFlowAndImage(Mat& flow, Mat& gray, Mat& out) {
	CV_Assert(gray.channels() == 1);
	if (!flow.empty()) {

		if(out.empty()){
			out = Mat(flow.rows, flow.cols, flow.type());
		}

		Mat flow_split[2];
		Mat magnitude, angle;
		Mat hsv_split[3], hsv;
		split(flow, flow_split);
		cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
		normalize(magnitude, magnitude, 0, 255, NORM_MINMAX);
		normalize(angle, angle, 0, 255, NORM_MINMAX);

		hsv_split[0] = angle; // already in degrees - no normalization needed
		Mat x;
		if(gray.empty()){
			x = Mat::ones(angle.size(), angle.type());
		} else{
			gray.convertTo(x, angle.type());
		}
		hsv_split[1] = x.clone();
		hsv_split[2] = magnitude;
		merge(hsv_split, 3, hsv);
		cvtColor(hsv, out, COLOR_HSV2BGR);
		normalize(out, out, 0, 255, NORM_MINMAX); // Normalise the matrix in the 0 - 255 range

		Mat n;
		out.convertTo(n, CV_8UC3); // Convert to 3 channel uchar matrix
		n.copyTo(out);

		// Normalise the flow within the range 0 ... 1
		normalize(flow, flow, 0, 1, NORM_MINMAX);
	}
}

void printStats(String folder, map<int32_t, vector<int32_t> > stats){
	ofstream myfile;
	String f = folder;
	String name = "/stats.csv";
	f += name;
	myfile.open(f.c_str());

	myfile << "Frame #,Sample Size,Selected Sample,Feature Size, Selected Features, # Clusters,Cluster Sum, Cluster Avg., Actual\n";

	for(map<int32_t, vector<int32_t> >::iterator it = stats.begin(); it != stats.end(); ++it){
		vector<int32_t> vv = it->second;
		myfile << it->first << ",";

		for(uint i = 0; i < vv.size(); ++i){
			myfile << vv[i] << ",";
		}
		myfile << "\n";
	}

	myfile.close();

}



void printClusterEstimates(String folder, map<int32_t, vector<int32_t> > cEstimates){
	ofstream myfile;
	String f = folder;
	String name = "/ClusterEstimates.csv";
	f += name;
	myfile.open(f.c_str());

	myfile << "Frame #,Cluster Sum, Cluster Avg.\n";

	for(map<int32_t, vector<int32_t> >::iterator it = cEstimates.begin(); it != cEstimates.end(); ++it){
		vector<int32_t> vv = it->second;
		size_t sz = vv.size();
		myfile << it->first << "," << vv[sz-1] << "," << vv[sz-2] << ",";

		for(uint i = 0; i < vv.size()-2; ++i){
			myfile << vv[i] << ",";
		}
		myfile << "\n";
	}

	myfile.close();

}


int main(int argc, char** argv) {
	//dummy_tester();
	ocl::setUseOpenCL(true);
	Mat frame;
	Ptr<Feature2D> detector;
	Ptr<GraphSegmentation> graphSegmenter = createGraphSegmentation();
	Ptr<Tracker> tracker = Tracker::create("BOOSTING");
	String destFolder;
	bool print = false;
	VideoCapture cap;
    Ptr<DenseOpticalFlow> algorithm;
    algorithm = optflow::createOptFlow_Farneback();
    BoxExtractor box;
    bool roiExtracted = false;
    int frameCount = 0;
    vocount vcount;

	cv::CommandLineParser parser(argc, argv, "{help ||}{dir||}{n|1|}"
			"{v||}{video||}");

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

	if (tracker == NULL) {
		cout << "***Error in the instantiation of the tracker...***\n";
		return -1;
	}

    detector = SURF::create(100);

    if( !cap.isOpened() ){
        printf("Could not open stream\n");
    	return -1;
    }


    for(;;)
    {

		bool read = cap.read(frame);

		if (read) {
			framed f;
			vector<Mat> d1;
			vector<int32_t> odata;

			f.i = frameCount;
			display("frame", frame);
			f.frame = frame.clone();

			cvtColor(f.frame, f.gray, COLOR_BGR2GRAY);
			vector<Mat> dataset;
			split(frame, dataset);
			int32_t estimation;

			d1.push_back(dataset[0]);
			d1.push_back(dataset[1]);
			d1.push_back(dataset[2]);

			if (vcount.frameHistory.size() > 0 && !f.gray.empty()) {

				Mat prevgray = vcount.frameHistory[vcount.frameHistory.size() - 2].gray;
				algorithm->calc(prevgray, f.gray, f.flow);
				Mat iFlow, fi;
				mergeFlowAndImage(f.flow, fi, iFlow);
				dataset.clear();
				pyrMeanShiftFiltering(iFlow, iFlow, 10, 30, 1);
				display("iFlow", iFlow);
				split(iFlow, dataset);
				d1.push_back(dataset[0]);
				d1.push_back(dataset[2]);
				//cout << "Getting d1 with flow" << endl;

			}

			//Mat mm;
			merge(d1, f.dataset);
			graphSegmenter->processImage(f.dataset, f.segments);

			if (!f.flow.empty()) {

				Mat nm;
				mergeFlowAndImage(f.flow, f.gray, nm);
				//display("nm", nm);
			}

			//printf("Rows before: %d\n", descriptors.rows);
			map<uint, vector<Point> > points;
			Mat output_image = getSegmentImage(f.segments, points);
			display("output_image", output_image);

			//printf("Points has size %d\n ", points.size());
			//Mat desc, fdesc;
			//vector<KeyPoint> kp;
			//pyrMeanShiftFiltering(frame, frame, 10, 30, 1);
			detector->detectAndCompute(frame, Mat(), f.keypoints, f.descriptors);
			//fdesc = desc.clone();
			uint ogsize = f.descriptors.rows;

			if (!roiExtracted && vcount.roiDesc.size() < 1) {
				Mat f2 = frame.clone();
				//drawKeypoints(output_image, kp, f2, Scalar::all(-1),
						//DrawMatchesFlags::DEFAULT);
				Mat x1;
				drawKeypoints(frame, f.keypoints, x1, Scalar::all(-1),
										DrawMatchesFlags::DEFAULT);

				display("x1", x1);
				//display("f2", f2);
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
				Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
						rng.uniform(0, 255));
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

						xp.push_back(f.keypoints[i]);
					}

				}
				//keypoints.push_back(roiPts);
				//descriptors.push_back(roiDesc);
				//printf("found %d object keypoints\n", roiDesc.rows);

				//printf("roiPts.size(): %d ----- xp.size(): %d\n", roiPts.size(), xp.size());
				//return 0;
			}

			if (!f.descriptors.empty()) {
				int32_t selectedSampleSize = 0;
				// Create clustering dataset
				Mat dset = f.descriptors.clone();

				for (uint n = 0; n < vcount.roiDesc.size(); ++n) {
					dset.push_back(vcount.roiDesc[n]);
				}

				hdbscan scan(dset, _EUCLIDEAN, 4, 4);
				scan.run();
				int x = scan.getClusterLabels().size();
				int y = dset.rows;
				printf("label size = %d\n and dataset is %d\n", x, y);
				getMappedPoint(f, scan);

				/********************************************************************
				 * Approximation of the number of similar objects
				 *******************************************************************/

				getCount(f, scan, ogsize);

				//vector<Mat> img_keypoints;

				Mat img_allkps = drawKeyPoints(frame, f.matchedKeypoints, Scalar(0, 0, 255));
				//Mat img_allkps ;
				//drawKeypoints(frame, allkps, img_allkps, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
				set<int> tempSet;
				tempSet.insert(f.labels.begin() + ogsize, f.labels.end());
				if(selectedSampleSize > 0){
					//printf("selectedSampleSize = %d\n", selectedSampleSize);
					estimation = (int32_t)f.total/selectedSampleSize;
					printf("This final approximation is %f\n", f.total);
				}
				cout << "Cluster " << f.largest << " is the largest" << endl;
				set<int> clusterSegments;
				for (uint x = 0; x < ogsize; ++x) {
					// keypoint at location i
					Point p;
					p.x = f.keypoints[x].pt.x;
					p.y = f.keypoints[x].pt.y;

					// label for the keypoint
					int label = f.labels[x];

					// find if the label is one of the query segment clusters
					set<int>::iterator fit = find(tempSet.begin(), tempSet.end(),
							label);

					if (fit != tempSet.end() && label != 0) {
						// the segment this keypoint is in
						float segment = f.segments.at<int>(p);
						clusterSegments.insert(segment);
						//printf("(%d, %d) label %d and segment %f\n", (int)p.x, (int)p.y, label, segment);
					}
				}

				printf("\n\n\n clusterSegments.size() : %d\n\n\n: ",
						clusterSegments.size());
				double min, max;
				minMaxLoc(f.segments, &min, &max);
				printf("Max segment is %f\n", max);


				cout
						<< "--------------------------------------------------------------------------------------------"
						<< endl;
				cout
						<< "---------------------------------------Statistics-------------------------------------------"
						<< endl;
				//cout << "Number of matchedKeyPoints is "
						//<< selectedFeatures << " out of " << ogsize
						//<< endl;
				cout
						<< "--------------------------------------------------------------------------------------------"
						<< endl;
				//printf("lset size = %d, stabilities size = %d\n", lset.size(), stabilities.size());
				//plotClusters2(fdesc, clusterSegments, labels);
				display("keypoints frame", f.keyPointImages[f.keyPointImages.size()-1]);
				//display("frame", frame);

				if (print && selectedSampleSize > 0) {
					printImage(destFolder, frameCount, "frame", frame);

					printImage(destFolder, frameCount, "output_image", output_image);
					Mat ff = drawKeyPoints(frame, f.keypoints, Scalar(0, 0, 255));
					printImage(destFolder, frameCount, "frame_kp", ff);

					for (uint i = 0; i < f.keyPointImages.size(); ++i) {
						string s = to_string(i);
						String ss = "img_keypoints-";
						ss += s.c_str();
						printImage(destFolder, frameCount, ss,
								f.keyPointImages[i]);
					}

					printImage(destFolder, frameCount, "img_allkps", img_allkps);

					odata.push_back(f.descriptors.rows);
					odata.push_back(selectedSampleSize);
					odata.push_back(ogsize);
					//odata.push_back(selectedFeatures);
					odata.push_back(f.keyPointImages.size());
					odata.push_back(f.total);
					int32_t avg = f.total/f.keyPointImages.size();
					odata.push_back(avg);
					odata.push_back(0);
					pair<int32_t, vector<int32_t> > pp(frameCount, odata);
					f.stats.insert(pp);
					//cest.push_back(avg);
					//cest.push_back(f.total);
					//pair<int32_t, vector<int32_t> > cpp(frameCount, cest);
					//clusterEstimates.insert(cpp);
				}
			}

			//printf("Rows after: %d\n", descriptors.rows);
			//cvtColor(f.image, f.gray, COLOR_BGR2GRAY);

			char c = (char) waitKey(20);
			if (c == 'q')
				break;
			maintaintHistory(vcount, f);
			++frameCount;
		} else{
			break;
		}
	}

    if(print){
    	framed fd = vcount.frameHistory[vcount.frameHistory.size()-1];
    	printStats(destFolder, fd.stats);
    	printClusterEstimates(destFolder, fd.clusterEstimates);
    }

	return 0;
}


