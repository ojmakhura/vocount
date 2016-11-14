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
using namespace hdbscan;

#define SAMPLE_SIZE	5

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

UndirectedGraph constructMST(Mat& src) {

	int nb_edges = 0;
	Mat dataset = src.clone();
	//int numPoints = dataset.rows * dataset.cols;
	//vector<int> clusterLabels = new vector<int>(numPoints, 0);
	int nb_channels = dataset.channels();

	int selfEdgeCapacity = 0;
	uint size = dataset.rows * dataset.cols;

	//One bit is set (true) for each attached point, or unset (false) for unattached points:
	//bool attachedPoints[selfEdgeCapacity] = { };
	//The MST is expanded starting with the last point in the data set:
	//unsigned int currentPoint = size - 1;
	//int numAttachedPoints = 1;
	//attachedPoints[size - 1] = true;

	//Each point has a current neighbor point in the tree, and a current nearest distance:
	vector<int>* nearestMRDNeighbors = new vector<int>(size);
	vector<float>* weights = new vector<float>(size, numeric_limits<float>::max());

	//Create an array for vertices in the tree that each point attached to:
	vector<int>* otherVertexIndices = new vector<int>(size);

	for (int i = 0; i < dataset.rows; i++) {
		const float* p = dataset.ptr<float>(i);

		for (int j = 0; j < dataset.cols; j++) {

			//Mat ds(Size(3,3), CV_32FC1, numeric_limits<float>::max());
			//vector<float> dsv(9, numeric_limits<float>::max());

			int from = i * dataset.rows + j;
			int to;
			int r = 0, c = 0, min_r = i, min_c = j;
			float min_d = numeric_limits<float>::max();
			float coreDistance = 0;
			min_d = numeric_limits<float>::max();

			for(int k = -1; k <= 1; ++k){
				for(int l = -1; l <= 1; ++l){

					float distance = 0;
					r = i+k;
					c = j+l;
					const float* p2 = dataset.ptr<float>(r);

					if(r >= 0 && c >= 0 && r < dataset.rows && c < dataset.cols && !(r == i && c == j)){
						//cout << "calculating distance " << endl;
						distance = 0;

						// calculate the distance between (i,j) and (k,l)
						for (int channel = 0; channel < nb_channels; channel++) {
							//printf("channel %d : (%f, %f)\n", channel, p[j * nb_channels + channel], p2[c * nb_channels + channel]);
							distance += pow(p[j * nb_channels + channel] - p2[c * nb_channels + channel], 2);
						}

						distance = sqrt(distance);

						if(distance > coreDistance){
							coreDistance = distance;
						}

						if(distance < min_d){

							//int from = i * dataset.rows + j;
							to = r * dataset.rows + c;
							min_d = distance;

							// If this edge does not exist
							if((*nearestMRDNeighbors)[to] != from){
								min_r = r;
								min_c = c;
							}
						}
					} else if(r == i && c == j){
						distance = 0;
					}
				}
			}
			(*otherVertexIndices)[from] = from;
			(*nearestMRDNeighbors)[from] = to;
			(*weights)[from] = min_d;
		}
	}
	return UndirectedGraph(size, *nearestMRDNeighbors, *otherVertexIndices, *weights);
	//printf("Dimensions: (%d, %d) -> \n", dataset.rows, dataset.cols);
	//cout << "Printing graph" << endl;
	//mst->print();
	//cout << "Done" << endl;
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

int main(int argc, char** argv) {
	int count = 0;
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

	if (tracker == NULL) {
		cout << "***Error in the instantiation of the tracker...***\n";
		return -1;
	}

    detector = SURF::create(100);

    VideoCapture cap(argv[1]);
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

    for(;;)
    {
    	Mat lap;
    	vector<Mat> segment_hist;
    	vector<float> segment_height, segment_width;
    	// Dataset made up of the segment histogram comparison with the {SAMPLE_SIZE}
    	// sample histograms and the with and height of the bounding boxes (hence the + 2)
    	vector<vector<float> > dataset(SAMPLE_SIZE+2);

        cap >> frame;
    	frame.copyTo(image);
        graphSegmenter->processImage(frame, gs);
        //printf("Rows before: %d\n", descriptors.rows);
		map<uint, vector<Point> > points;
        Mat output_image = getSegmentImage(gs, points);
        display("output_image", output_image);

        printf("Points has size %d\n ", points.size());
    	Mat desc;
    	vector<KeyPoint> kp;
        //pyrMeanShiftFiltering(frame, frame, 10, 30, 1);
        detector->detectAndCompute(frame, Mat(), kp, desc);
        uint ogsize = desc.rows;

        if(!roiExtracted && descriptors.size() < 1){
        	roi = box.extract("track", frame);

	        //initializes the tracker
	        if( !tracker->init( frame, roi ) )
	        {
	          cout << "***Could not initialize tracker...***\n";
	          return -1;
	        }

			roiExtracted = true;

        } else if(descriptors.size() < SAMPLE_SIZE){
        	tracker->update(frame, roi);
        	Scalar value = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
        	rectangle(frame, roi, value, 2, 8, 0);
        }

		// add query descriptors and keypoints to the query vectors
		if (base_hist.size() < SAMPLE_SIZE) {
		    vector<Point2f> roiPts;
		    Mat roiDesc;
	        // Get all keypoints inside the roi
			for(uint i = 0; i < kp.size(); ++i){
				Point p;
				p.x = kp[i].pt.x;
				p.y = kp[i].pt.y;

				if(roi.contains(kp[i].pt)){
					roiPts.push_back(kp[i].pt);
					roiDesc.push_back(desc.row(i));
				}
			}
			keypoints.push_back(kp);
			descriptors.push_back(roiDesc);

			// Calculate histograms for each tracked position and store them for later
			Mat h = getHist(frame, roi);
			base_hist.push_back(h);

			base_height.push_back(roi.height);
			base_width.push_back(roi.width);

			// The first 5 entries of the histogram dataset are made up of
			//
		} else {
			// Create datapoints from the samples and
			if(sample_dataset[0].size() == 0){ // only do this if it hasn't been done before
				for(uint i = 0; i < SAMPLE_SIZE; ++i){
					for(uint j = 0; j < SAMPLE_SIZE; ++j){
						double compare = compareHist(base_hist[i], base_hist[j], CV_COMP_CORREL);
						sample_dataset[i][j] = compare;

						dataset[i].push_back(compare);
					}
					sample_dataset[SAMPLE_SIZE].push_back(base_height[i]);
					dataset[SAMPLE_SIZE].push_back(base_height[i]);

					sample_dataset[SAMPLE_SIZE+1].push_back(base_width[i]);
					dataset[SAMPLE_SIZE+1].push_back(base_width[i]);
				}
			} else {
				// Just copy the data already in the sample_dataset
				// TODO: Do thi as the end of composing the dataset. For ease of indexing the results.
				for (uint i = 0; i < SAMPLE_SIZE; ++i) {
					dataset[i].insert(dataset[i].begin(), sample_dataset[i].begin(), sample_dataset[i].end());
				}
			}
		}

        //printf("Rows after: %d\n", descriptors.rows);
        image = frame.clone();
        cvtColor(image, gray, COLOR_BGR2GRAY);

        if( !prevgray.empty() && base_hist.size() == SAMPLE_SIZE)
        {
			absdiff(frame, _prev, fdiff);

            //algorithm->calc(gray, prevgray, uflow);
            cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
            normalize(cflow, cflow, 0, 255, NORM_MINMAX);
            //uflow.copyTo(flow);
            //display("myflow", flow);

            drawKeypoints( frame, kp, keyPointImage, Scalar::all(-1), DrawMatchesFlags::DEFAULT );

            // TODO: calculate histograms and size of segments and use them to create the dataset
            for (std::map<uint,vector<Point> >::iterator it=points.begin(); it!=points.end(); ++it){
            	vector<Point> ps = it->second;
            	roi = boundingRect(ps);

            	Mat m = frame(roi);
            	Mat h = getHist(frame, roi);

            	for(uint i = 0; i < SAMPLE_SIZE; ++i){
            		double compare = compareHist(h, base_hist[i], CV_COMP_CORREL);
            		dataset[i].push_back(compare);
            	}

            	dataset[SAMPLE_SIZE].push_back(roi.height);
            	dataset[SAMPLE_SIZE].push_back(roi.width);
            }
            //display("frame", frame;
            // TODO: Run HDBSCAN on the dataset
        }
        display("frame", frame);

        if(waitKey(2000)>=0)
            break;
        std::swap(prevgray, gray);
        std::swap(_prev, frame);
        ++count;
    }

	return 0;
}

void extras(){

	/*
	 if(!desc.empty()){

                // Create clustering dataset
                for(uint n = 0; n < descriptors.size(); ++n){
                	desc.push_back(descriptors[n]);
                }


				map<int, vector<Point> > mappedPoints;

				HDBSCAN scan(desc, _EUCLIDEAN, 3, 3);
				//printf("scan creation done\n");
				scan.run(true);
				//printf("scan cluster done\n");
				vector<int> labels = scan.getClusterLabels();
				map<int, float> stabilities = scan.getClusterStabilities();
				set<int> lset;
				vector<float> sstabilities(lset.size());
				lset.insert(labels.begin(), labels.end());
				//int i = 0;
				for(set<int>::iterator it = lset.begin(); it != lset.end(); ++it){
					vector<Point> pts;
					for(uint i = 0; i < labels.size(); ++i){
						if(*it == labels[i]){
							Point p;
							p.x = (int)kp[i].pt.x;
							p.y = (int)kp[i].pt.y;

							pts.push_back(p);
						}
					}
					//printf("%d has %d\n\t\n\n\n", *it, pts.size());
					//foruint i = 0; i < pts.size())
					pair<uint, vector<Point> > pr(*it, pts);
					mappedPoints.insert(pr);
					//++i;
					Scalar value = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
					Rect rr = boundingRect(pts);
					//rectangle(keyPointImage, rr, value, 2, 8, 0);
					string name = to_string(*it);
					putText(keyPointImage, name.c_str(), Point(rr.x - 4, rr.y - 4), CV_FONT_HERSHEY_PLAIN, 1, value);
				}

				*******************************************************************
				 * Approximation of the number of similar objects
				 *******************************************************************

				map<int, vector<int> > roiClusters;

				//int i = 0;
				int add_size = 0;
				for(set<int>::iterator it = lset.begin(); it != lset.end(); ++it){
					vector<int> pts;
					for(uint i = ogsize; i < labels.size(); ++i){
						if(*it == labels[i] && *it != 0){
							pts.push_back(i);
						}
					}

					if(!pts.empty()){
						float n = (float)mappedPoints[*it].size()/pts.size();
						printf("stability: %f --> %d has %d and total is %d :: Approx Num of objects: %f\n\n", stabilities[*it], *it, pts.size(), mappedPoints[*it].size(), n);
						pair<uint, vector<int> > pr(*it, pts);
						roiClusters.insert(pr);
					}
				}
				cout << "--------------------------------------------------------------------------------------------" << endl;
				cout << "---------------------------------------Statistics-------------------------------------------" << endl;
				cout << "--------------------------------------------------------------------------------------------" << endl;
				//printf("lset size = %d, stabilities size = %d\n", lset.size(), stabilities.size());

				display("keypoints frame", keyPointImage);
				//display("frame", frame);
            }
	 */

}
