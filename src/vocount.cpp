#include "hdbscan.hpp"
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

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ximgproc::segmentation;

vector<vector<Point> > contours;
vector<Vec4i> hierarchy;
Mat _prev, prevgray, fdiff, 	// previous two frames
		 diff, 	// previous image differences
		 edge, 	// previous edge difference edges //fEdge[2],
		 flow, cflow, uflow,		// Optical flow as a colour image
		 frame,		// image version of flow
		gray,	// previous optical flows
		 image,		// the current image
		 cImage,	// contour image
		 candidates,// candidates computed from the contours
		 iEdge;		// edges for the current image
RNG rng(12345);

int SAMPLE_SIZE = 1;

void display(char const* screen, const InputArray& m) {
	if (!m.empty()) {
		namedWindow(screen, WINDOW_AUTOSIZE);
		imshow(screen, m);
	}
}

Scalar hsv_to_rgb(Scalar c) {
    Mat in(1, 1, CV_32FC3);
    Mat out(1, 1, CV_32FC3);

    float * p = in.ptr<float>(0);

    p[0] = c[0] * 360;
    p[1] = c[1];
    p[2] = c[2];

    cvtColor(in, out, COLOR_HSV2RGB);

    Scalar t;

    Vec3f p2 = out.at<Vec3f>(0, 0);

    t[0] = (int)(p2[0] * 255);
    t[1] = (int)(p2[1] * 255);
    t[2] = (int)(p2[2] * 255);

    return t;

}

Scalar color_mapping(int segment_id) {

    double base = (double)(segment_id) * 0.618033988749895 + 0.24443434;

    return hsv_to_rgb(Scalar(fmod(base, 1.2), 0.95, 0.80));

}

Mat getSegmentImage(Mat& gs, map<uint, vector<Point> >& points){
	//map<uint, vector<Point> > points;
    Mat output_image = Mat::zeros(gs.rows, gs.cols, CV_8UC3);
    uint* p;
    uchar* p2;

    /**
     * Get the segmentation points that will be assigned to individual
     */
	for (int i = 0; i < gs.rows; i++) {
		p = gs.ptr<uint>(i);
		p2 = output_image.ptr<uchar>(i);
		for (int j = 0; j < gs.cols; j++) {
			pair<uint, vector<Point> > pr(p[j], vector<Point>());
			std::map<uint, vector<Point> >::iterator it;
			it = points.find(p[j]);

			if (it == points.end()) {
				pair<map<uint, vector<Point> >::iterator, bool> ret =
						points.insert(pr);
				ret.first->second.push_back(Point(j, i));
			} else {
				it->second.push_back(Point(j, i));
			}

			Scalar color = color_mapping(p[j]);
			p2[j * 3] = color[0];
			p2[j * 3 + 1] = color[1];
			p2[j * 3 + 2] = color[2];
		}
	}

	return output_image;
}

Mat getHist(Mat frame, Rect roi){
	Mat hsv;
	Mat m = frame(roi);
	cvtColor(m, hsv, CV_BGR2HSV);
	int h_bins = 50;
	int s_bins = 60;
	int histSize[] = { h_bins, s_bins };

	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };

	const float* ranges[] = { h_ranges, s_ranges };

	int channels[] = { 0, 1 };
	Mat hist;
	calcHist(&hsv, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
	normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

	return hist;
}

void printImage(String folder, int idx, String name, Mat img) {
	//string folder = "/home/ojmakh/programming/phd/data/";
	stringstream sstm;

	sstm << folder.c_str() << "/" << idx << " " << name.c_str() << ".jpg";
	//cout << "printing " << sstm.str() << endl;
	imwrite(sstm.str(), img);
}

/**
 * Find the possible segment of interest
 * This function works by looking at the segments on both
 * sides of the roi rectangle. If the same segment is on
 * both sides, then it is added to the list of segments to
 * ignore.
 */
set<int32_t> getIgnoreSegments(Rect roi, Mat segments){
	set<int32_t> span;
	int32_t t = 3;
	cout << roi << endl;
	printf("roi.height = %d : roi.width = %d\n", roi.height, roi.width);
	printf("roi.x = %d : roi.y = %d\n", roi.x, roi.y);

	for(int i = roi.x; i < roi.x+roi.width; ++i){
		Point p1(roi.x, roi.y-t);
		Point p2(roi.x, roi.y+t);

		int32_t i1 = segments.at<int32_t>(p1);
		int32_t i2 = segments.at<int32_t>(p2);

		if(i1 == i2){
			span.insert(i1);
		}

		///printf("(p1.x, p1.y) = (%d, %d) ::::::: (p2.x, p2.y) = (%d, %d)\n", p1.x, p2.y, i1, i2);

		int end = roi.y + roi.height;
		p1.y = end + t;
		p2.y = end - t;

		i1 = segments.at<int32_t>(p1);
		i2 = segments.at<int32_t>(p2);

		if(i1 == i2){
			span.insert(i1);
		}
		///printf("(p1.x, p1.y) = (%d, %d) ::::::: (p2.x, p2.y) = (%d, %d)\n", p1.x, p2.y, i1, i2);
	}

	for(int j = roi.y; j < roi.y + roi.height; ++j){
		Point p1(roi.x - t, roi.y);
		Point p2(roi.x + t, roi.y);

		int32_t i1 = segments.at<int32_t>(p1);
		int32_t i2 = segments.at<int32_t>(p2);

		if(i1 == i2){
			span.insert(i1);
		}
		///printf("(p1.x, p1.y) = (%d, %d) ::::::: (p2.x, p2.y) = (%d, %d)\n", p1.x, p2.y, i1, i2);

		int end = roi.x + roi.width;
		p1.x = end + t;
		p2.x = end - t;

		i1 = segments.at<int32_t>(p1);
		i2 = segments.at<int32_t>(p2);

		if(i1 == i2){
			span.insert(i1);
		}
		///printf("(p1.x, p1.y) = (%d, %d) ::::::: (p2.x, p2.y) = (%d, %d)\n", p1.x, p2.y, i1, i2);

	}

	printf("span has %d\n", span.size());

	return span;
}

Mat drawKeyPoints(Mat in, vector<KeyPoint> points, Scalar colour){
	Mat x = in.clone();

	for(vector<KeyPoint>::iterator it = points.begin(); it != points.end(); ++it){
		circle(x, Point(it->pt.x, it->pt.y), 4, colour, CV_FILLED, 8, 0);
	}


	return x;
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
	vector<vector<KeyPoint> > keypoints;
	Mat keyPointImage;
	vector<Mat> descriptors;
	vector<Mat> base_hist, segment_hist;
	vector<float> base_height, segment_height, base_width, segment_width;
	Ptr<Feature2D> detector;
	Ptr<GraphSegmentation> graphSegmenter = createGraphSegmentation();
	Ptr<Tracker> tracker = Tracker::create("BOOSTING");
	String destFolder;
	bool print = false;
	VideoCapture cap;

	int compare_method = CV_COMP_CORREL;

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

    Ptr<DenseOpticalFlow> algorithm;
    algorithm = optflow::createOptFlow_Farneback();
    BoxExtractor box;
    bool roiExtracted = false;
    Rect2d roi;
    Mat gs;
    int frameCount = 0;
    vector<vector<float> > sample_dataset(SAMPLE_SIZE+2);
    map<int32_t, vector<int32_t> > stats;
    map<int32_t, vector<int32_t> > clusterEstimates;

    for(;;)
    {
		Mat fkprint;
		//vector<Mat> segment_hist;
		vector<float> segment_height, segment_width;
		// Dataset made up of the segment histogram comparison with the {SAMPLE_SIZE}
		// sample histograms and the with and height of the bounding boxes (hence the + 2)
		//vector<vector<float> > dataset(SAMPLE_SIZE+2);

		//cap >> frame;
		bool read = cap.read(frame);

		if (read) {
			display("frame", frame);
			vector<int32_t> odata;

			frame.copyTo(image);
			cvtColor(frame, gray, COLOR_BGR2GRAY);
			vector<Mat> dataset;
			split(frame, dataset);
			vector<Mat> d1;
			int32_t estimation;

			d1.push_back(dataset[0]);
			d1.push_back(dataset[1]);
			d1.push_back(dataset[2]);

			if (!prevgray.empty() && !gray.empty()) {
				algorithm->calc(prevgray, gray, flow);
				Mat iFlow, f;
				mergeFlowAndImage(flow, f, iFlow);
				dataset.clear();
				pyrMeanShiftFiltering(iFlow, iFlow, 10, 30, 1);
				display("iFlow", iFlow);
				split(iFlow, dataset);
				d1.push_back(dataset[0]);
				d1.push_back(dataset[2]);
				//cout << "Getting d1 with flow" << endl;

			} else{
				prevgray = gray.clone();
				continue;
			}
			Mat mm;
			merge(d1, mm);
			graphSegmenter->processImage(mm, gs);

			if (!flow.empty()) {

				Mat nm;
				mergeFlowAndImage(flow, gray, nm);
				//display("nm", nm);
			}

			//printf("Rows before: %d\n", descriptors.rows);
			map<uint, vector<Point> > points;
			Mat output_image = getSegmentImage(gs, points);
			display("output_image", output_image);

			//printf("Points has size %d\n ", points.size());
			Mat desc, fdesc;
			vector<KeyPoint> kp;
			//pyrMeanShiftFiltering(frame, frame, 10, 30, 1);
			detector->detectAndCompute(frame, Mat(), kp, desc);
			fdesc = desc.clone();
			uint ogsize = desc.rows;

			if (!roiExtracted && descriptors.size() < 1) {
				Mat f2 = frame.clone();
				//drawKeypoints(output_image, kp, f2, Scalar::all(-1),
						//DrawMatchesFlags::DEFAULT);
				Mat x1;
				drawKeypoints(frame, kp, x1, Scalar::all(-1),
										DrawMatchesFlags::DEFAULT);

				display("x1", x1);
				//display("f2", f2);
				roi = box.extract("Select ROI", f2);

				//initializes the tracker
				if (!tracker->init(frame, roi)) {
					cout << "***Could not initialize tracker...***\n";
					return -1;
				}

				roiExtracted = true;

			} else if (descriptors.size() < SAMPLE_SIZE) {
				tracker->update(frame, roi);
				Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
						rng.uniform(0, 255));
				rectangle(frame, roi, value, 2, 8, 0);
				vector<KeyPoint> roiPts, xp;
				Mat roiDesc;
				set<int32_t> ignore = getIgnoreSegments(roi, gs);

				/*for(set<int32_t>::iterator it = ignore.begin(); it != ignore.end(); ++it){
				 printf("Segment : %d\n\n", *it);
				 }*/

				// Get all keypoints inside the roi
				for (uint i = 0; i < kp.size(); ++i) {
					Point p;
					p.x = kp[i].pt.x;
					p.y = kp[i].pt.y;

					int32_t seg = gs.at<int32_t>(p); // get the segmentation id at point p

					// find if the segment id is listed in the ignore list
					set<int32_t>::iterator it = std::find(ignore.begin(),
							ignore.end(), seg);

					if (roi.contains(kp[i].pt)) { //&&
						//printf("Segment is %d \n\n", seg);
						if (it == ignore.end()) {
							roiPts.push_back(kp[i]);
							roiDesc.push_back(desc.row(i));
						}

						xp.push_back(kp[i]);
					}

				}
				keypoints.push_back(roiPts);
				descriptors.push_back(roiDesc);
				//printf("found %d object keypoints\n", roiDesc.rows);

				//printf("roiPts.size(): %d ----- xp.size(): %d\n", roiPts.size(), xp.size());
				//return 0;
			}

			if (!desc.empty()) {
				int32_t selectedSampleSize = 0;
				Mat d = desc.clone();
				// Create clustering dataset
				for (uint n = 0; n < descriptors.size(); ++n) {
					d.push_back(descriptors[n]);
				}

				map<int, vector<KeyPoint> > mappedPoints;
				//map<int, vector<KeyPoint> > kMappedPoints;

				hdbscan scan(_EUCLIDEAN, 4, 4);
				//printf("scan creation done\n");
				scan.run(d);
				//printf("scan cluster done\n");
				vector<int> labels = scan.getClusterLabels();
				map<int, float> stabilities = scan.getClusterStabilities();
				set<int> lset, tempSet;
				vector<float> sstabilities(lset.size());
				lset.insert(labels.begin(), labels.end());
				// temp set is the set of clusters for the sample data
				tempSet.insert(labels.begin() + ogsize, labels.end());
				//int i = 0;
				for (set<int>::iterator it = lset.begin(); it != lset.end();
						++it) {
					vector<KeyPoint> pts;
					for (uint i = 0; i < labels.size(); ++i) {
						if (*it == labels[i]) {
							Point p;
							p.x = (int) kp[i].pt.x;
							p.y = (int) kp[i].pt.y;

							pts.push_back(kp[i]);
						}
					}
					//printf("%d has %d\n\t\n\n\n", *it, pts.size());
					//foruint i = 0; i < pts.size())
					pair<uint, vector<KeyPoint> > pr(*it, pts);
					mappedPoints.insert(pr);

					//++i;
					Scalar value = Scalar(rng.uniform(0, 255),
							rng.uniform(0, 255), rng.uniform(0, 255));
					//Rect rr = boundingRect(pts);
					//rectangle(keyPointImage, rr, value, 2, 8, 0);
					//string name = to_string(*it);
					//putText(keyPointImage, name.c_str(), Point(rr.x - 4, rr.y - 4),
					//CV_FONT_HERSHEY_PLAIN, 1, value);
				}

				/********************************************************************
				 * Approximation of the number of similar objects
				 *******************************************************************/

				map<int, vector<int> > roiClusters;
				vector<Mat> img_keypoints;
				vector<int32_t> cest;
				vector<KeyPoint> allkps;
				uint largest = 0;
				float lsize = 0;
				int32_t selectedFeatures = 0;
				//int i = 0;
				int add_size = 0;
				float total = 0;
				cout << "################################################################################" << endl;
				cout << "                              " << frameCount << endl;
				cout << "################################################################################"<< endl;

				for (set<int>::iterator it = lset.begin(); it != lset.end();
						++it) {
					vector<int> pts;
					for (uint i = ogsize; i < labels.size(); ++i) {
						if (*it == labels[i] && *it != 0) {
							pts.push_back(i);
							selectedSampleSize++;
						}
					}

					if (!pts.empty()) {
						int32_t n = mappedPoints[*it].size() / pts.size();
						total += n;
						printf(
								"stability: %f --> %d has %d and total is %d :: Approx Num of objects: %d\n\n",
								stabilities[*it], *it, pts.size(),
								mappedPoints[*it].size(), n);
						pair<uint, vector<int> > pr(*it, pts);
						roiClusters.insert(pr);
						selectedFeatures += mappedPoints[*it].size();
						cest.push_back(n);

						if (n > lsize) {
							largest = *it;
							lsize = n;
						}
						Mat kimg = drawKeyPoints(frame, mappedPoints[*it], Scalar(0, 0, 255));;
						///drawKeypoints(frame, mappedPoints[*it], kimg,
								//Scalar::all(-1), DrawMatchesFlags::DEFAULT);

						img_keypoints.push_back(kimg);
						allkps.insert(allkps.end(), mappedPoints[*it].begin(), mappedPoints[*it].end());
					}
				}
				Mat img_allkps = drawKeyPoints(frame, allkps, Scalar(0, 0, 255));
				//Mat img_allkps ;
				//drawKeypoints(frame, allkps, img_allkps, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

				if(selectedSampleSize > 0){
					//printf("selectedSampleSize = %d\n", selectedSampleSize);
					estimation = (int32_t)total/selectedSampleSize;
					printf("This final approximation is %f\n", total);
				}
				cout << "Cluster " << largest << " is the largest" << endl;
				//cout << gs << endl;
				set<int> clusterSegments;
				for (uint x = 0; x < ogsize; ++x) {
					// keypoint at location i
					Point p;
					p.x = kp[x].pt.x;
					p.y = kp[x].pt.y;

					// label for the keypoint
					int label = labels[x];

					// find if the label is one of the query segment clusters
					set<int>::iterator f = find(tempSet.begin(), tempSet.end(),
							label);

					if (f != tempSet.end() && label != 0) {
						// the segment this keypoint is in
						float segment = gs.at<int>(p);
						clusterSegments.insert(segment);
						//printf("(%d, %d) label %d and segment %f\n", (int)p.x, (int)p.y, label, segment);
					}
				}

				printf("\n\n\n clusterSegments.size() : %d\n\n\n: ",
						clusterSegments.size());
				double min, max;
				minMaxLoc(gs, &min, &max);
				printf("Max segment is %f\n", max);


				cout
						<< "--------------------------------------------------------------------------------------------"
						<< endl;
				cout
						<< "---------------------------------------Statistics-------------------------------------------"
						<< endl;
				cout << "Number of matchedKeyPoints is "
						<< selectedFeatures << " out of " << ogsize
						<< endl;
				cout
						<< "--------------------------------------------------------------------------------------------"
						<< endl;
				//printf("lset size = %d, stabilities size = %d\n", lset.size(), stabilities.size());
				//plotClusters2(fdesc, clusterSegments, labels);
				display("keypoints frame", keyPointImage);
				//display("frame", frame);

				if (print && selectedSampleSize > 0) {
					printImage(destFolder, frameCount, "frame", frame);

					printImage(destFolder, frameCount, "output_image", output_image);
					Mat ff = drawKeyPoints(frame, kp, Scalar(0, 0, 255));
					printImage(destFolder, frameCount, "frame_kp", ff);

					for (uint i = 0; i < img_keypoints.size(); ++i) {
						string s = to_string(i);
						String ss = "img_keypoints-";
						ss += s.c_str();
						printImage(destFolder, frameCount, ss,
								img_keypoints[i]);
					}

					printImage(destFolder, frameCount, "img_allkps", img_allkps);

					odata.push_back(descriptors[0].rows);
					odata.push_back(selectedSampleSize);
					odata.push_back(ogsize);
					odata.push_back(selectedFeatures);
					odata.push_back(img_keypoints.size());
					odata.push_back(total);
					int32_t avg = total/img_keypoints.size();
					odata.push_back(avg);
					odata.push_back(0);
					pair<int32_t, vector<int32_t> > pp(frameCount, odata);
					stats.insert(pp);
					cest.push_back(avg);
					cest.push_back(total);
					pair<int32_t, vector<int32_t> > cpp(frameCount, cest);
					clusterEstimates.insert(cpp);
				}
			}



			//printf("Rows after: %d\n", descriptors.rows);
			image = frame.clone();
			cvtColor(image, gray, COLOR_BGR2GRAY);

			char c = (char) waitKey(20);
			if (c == 'q')
				break;
			std::swap(prevgray, gray);
			std::swap(_prev, frame);
			++frameCount;
		} else{
			break;
		}
	}

    if(print){
    	printStats(destFolder, stats);
    	printClusterEstimates(destFolder, clusterEstimates);
    }

	return 0;
}


