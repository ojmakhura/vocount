/*
 * process_frame.cpp
 *
 *  Created on: 3 May 2017
 *      Author: ojmakh
 */

#include "process_frame.hpp"



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

void getMappedPoint(framed& f, hdbscan& scan){
	RNG rng(12345);
	f.labels = scan.getClusterLabels();
	map<int, float> stabilities = scan.getClusterStabilities();
	set<int> lset, tempSet;
	vector<float> sstabilities(lset.size());
	lset.insert(f.labels.begin(), f.labels.end());
	// temp set is the set of clusters for the sample data
	//int i = 0;
	for (set<int>::iterator it = lset.begin(); it != lset.end(); ++it) {
		vector<KeyPoint> pts;
		for (uint i = 0; i < f.labels.size(); ++i) {
			if (*it == f.labels[i]) {
				Point p;
				p.x = (int) f.keypoints[i].pt.x;
				p.y = (int) f.keypoints[i].pt.y;

				pts.push_back(f.keypoints[i]);
			}
		}
		//printf("%d has %d\n\t\n\n\n", *it, pts.size());
		//foruint i = 0; i < pts.size())
		pair<uint, vector<KeyPoint> > pr(*it, pts);
		f.mappedPoints.insert(pr);

		//++i;
		Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
				rng.uniform(0, 255));
		//Rect rr = boundingRect(pts);
		//rectangle(keyPointImage, rr, value, 2, 8, 0);
		//string name = to_string(*it);
		//putText(keyPointImage, name.c_str(), Point(rr.x - 4, rr.y - 4),
		//CV_FONT_HERSHEY_PLAIN, 1, value);
	}
}

void getCount(framed& f, hdbscan& scan, int ogsize){
	cout
			<< "################################################################################"
			<< endl;
	cout << "                              " << f.i << endl;
	cout
			<< "################################################################################"
			<< endl;

	set<int> lset;
	map<int, float> stabilities = scan.getClusterStabilities();
	lset.insert(f.labels.begin(),f. labels.end());
	for (set<int>::iterator it = lset.begin(); it != lset.end(); ++it) {
		vector<int> pts;
		for (uint i = ogsize; i < f.labels.size(); ++i) {
			if (*it == f.labels[i] && *it != 0) {
				pts.push_back(i);
			}
		}

		if (!pts.empty()) {
			int32_t n = f.mappedPoints[*it].size() / pts.size();
			f.total += n;
			printf(
					"stability: %f --> %d has %d and total is %d :: Approx Num of objects: %d\n\n",
					stabilities[*it], *it, pts.size(),
					f.mappedPoints[*it].size(), n);
			pair<uint, vector<int> > pr(*it, pts);
			f.roiClusters.insert(pr);

			if (n > f.lsize) {
				f.largest = *it;
				f.lsize = n;
			}
			Mat kimg = drawKeyPoints(f.frame, f.mappedPoints[*it],
					Scalar(0, 0, 255));
			;
			///drawKeypoints(frame, mappedPoints[*it], kimg,
			//Scalar::all(-1), DrawMatchesFlags::DEFAULT);

			f.keyPointImages.push_back(kimg);
			f.matchedKeypoints.insert(f.matchedKeypoints.end(), f.mappedPoints[*it].begin(),
					f.mappedPoints[*it].end());
		}
	}
}

void maintaintHistory(vocount& voc, framed& f){
	voc.frameHistory.push_back(f);
	if(voc.frameHistory.size() > 10){
		voc.frameHistory.erase(voc.frameHistory.begin());
	}
}

