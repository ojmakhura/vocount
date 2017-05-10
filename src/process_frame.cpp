/*
 * process_frame.cpp
 *
 *  Created on: 3 May 2017
 *      Author: ojmakh
 */

#include "process_frame.hpp"
#include <fstream>

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


	return span;
}

Mat drawKeyPoints(Mat in, vector<KeyPoint> points, Scalar colour, int type){
	Mat x = in.clone();
	if(type == -1){
		for(vector<KeyPoint>::iterator it = points.begin(); it != points.end(); ++it){
			circle(x, Point(it->pt.x, it->pt.y), 4, colour, CV_FILLED, 8, 0);
		}
	} else{
		drawKeypoints( in, points, x, Scalar::all(-1), type );
	}

	return x;
}


vector<KeyPoint> getAllMatchedKeypoints(framed& f){
	vector<KeyPoint> kp;

	for(map<int, int>::iterator it = f.roiClusterCount.begin(); it != f.roiClusterCount.end(); ++it){
		vector<KeyPoint> k = f.mappedKeyPoints[it->first];
		kp.insert(kp.end(), k.begin(), k.end());
	}

	return kp;
}

void mapKeyPoints(framed& f, hdbscan& scan, int ogsize){
	// Only labels from the first n indices where n is the number of features found in f.frame
	f.labels.insert(f.labels.begin(), scan.getClusterLabels().begin(), scan.getClusterLabels().begin()+f.descriptors.rows);

	// add the indices and keypoints using labels as the key to the map
	for(int i = 0; i < f.labels.size(); i++){
		int l = f.labels[i];
		f.mappedKeyPoints[l].push_back(f.keypoints[i]);
		f.mappedLabels[l].push_back(i);
	}

	// get a cluster labels belonging to the sample features and map them with the number of labels

	for(vector<int>::iterator it = scan.getClusterLabels().begin() + ogsize; it != scan.getClusterLabels().end(); ++it){
		if(*it != 0){
			f.roiClusterCount[*it]++;
		}
	}

}

/**
 *
 */
void getCount(framed& f, hdbscan& scan, int ogsize){
	cout << "################################################################################" << endl;
	cout << "                              " << f.i << endl;
	cout << "################################################################################" << endl;

	map<int, float> stabilities = scan.getClusterStabilities();

	for (map<int, int>::iterator it = f.roiClusterCount.begin(); it != f.roiClusterCount.end(); ++it) {
		//vector<int> pts = f.mappedKeyPoints[*it];
		f.total += it->second;

		//if (!pts.empty()) {
		int32_t n = f.mappedKeyPoints[it->first].size() / it->second;
		f.total += it->second;
		printf(
				"stability: %f --> %d has %d and total is %d :: Approx Num of objects: %d\n\n",
				stabilities[it->first], it->first, it->second,
				f.mappedKeyPoints[it->first].size(), n);
		f.selectedFeatures += f.mappedKeyPoints[it->first].size();

		if (f.mappedKeyPoints[it->first].size() > f.lsize) {
			f.largest = it->first;
			f.lsize = f.mappedKeyPoints[it->first].size();
		}

		Mat kimg = drawKeyPoints(f.frame, f.mappedKeyPoints[it->first],
				Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

		f.keyPointImages.push_back(kimg);

	}
}

void maintaintHistory(vocount& voc, framed& f){
	voc.frameHistory.push_back(f);
	if(voc.frameHistory.size() > 10){
		voc.frameHistory.erase(voc.frameHistory.begin());
	}
}

void runSegmentation(vocount& vcount, framed& f, Ptr<GraphSegmentation> graphSegmenter, Ptr<DenseOpticalFlow> flowAlgorithm){
	vector<Mat> dataset;
	split(f.frame, dataset);
	int32_t estimation;
	vector<Mat> d1;

	d1.push_back(dataset[0]);
	d1.push_back(dataset[1]);
	d1.push_back(dataset[2]);

	if (vcount.frameHistory.size() > 0 && !f.gray.empty()) {
		framed fx = vcount.frameHistory[vcount.frameHistory.size() - 1];
		Mat prevgray = vcount.frameHistory[vcount.frameHistory.size() - 1].gray;
		flowAlgorithm->calc(prevgray, f.gray, f.flow);
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

Mat getDataset(vocount& vcount, framed& f, uint* ogsize){
	Mat dataset = f.descriptors;
	if (!vcount.frameHistory.empty()) {
		for (int j = 1; j < vcount.step; ++j) {
			int ix = vcount.frameHistory.size() - j;
			if (ix > 0) {
				framed fx = vcount.frameHistory[ix];
				dataset.push_back(fx.descriptors);
			}
		}
	}
	*ogsize = dataset.rows;
	for (uint n = 0; n < vcount.roiDesc.size(); ++n) {
		dataset.push_back(vcount.roiDesc[n]);
	}
	f.dataset = dataset;
	return dataset;
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

void matchByBruteForce(vocount& vcount, framed& f){
	BFMatcher matcher(NORM_L1);
	vector< DMatch > good_matches;
	vector< DMatch > matches;
    // drawing the results
    Mat img_matches;
	//matcher.add(vcount.roiDesc[0]);
	//matcher.train();
	matcher.match(f.descriptors, vcount.roiDesc[0], good_matches, Mat());
	//matcher.match(f.descriptors, good_matches, Mat());//-- Localize the object
	printf("train has %d, query has %d and good matches has %d\n", vcount.roiDesc[0].rows, f.descriptors.rows, good_matches.size());
    drawMatches( vcount.samples[0], vcount.roiKeypoints[0], f.frame, f.keypoints,
                 good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                 std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS  );
    display("img_matches", img_matches);
    /*std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( vcount.roiKeypoints[0][ good_matches[i].queryIdx ].pt );
        scene.push_back( f.keypoints[ good_matches[i].trainIdx ].pt );
    }*/
}

void matchByFLANN(vocount& vcount, framed& f){
	vector< DMatch > good_matches;
	vector<vector< DMatch > > matches;
	Mat objectMat = vcount.samples[0];
	Mat sceneMat = f.frame;
    //vector of keypoints
    vector< cv::KeyPoint > keypointsO = vcount.roiKeypoints[0];
    vector< cv::KeyPoint > keypointsS = f.keypoints;
    Mat descriptors_object = f.descriptors;
    Mat descriptors_scene = vcount.roiDesc[0];
	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	matcher.knnMatch(descriptors_object, descriptors_scene, matches, 2);
	good_matches.reserve(matches.size());

}
