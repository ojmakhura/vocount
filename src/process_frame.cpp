/*
 * process_frame.cpp
 *
 *  Created on: 3 May 2017
 *      Author: ojmakh
 */

#include "process_frame.hpp"
#include <fstream>
#include <opencv/cv.hpp>
#include <dirent.h>
#include <gsl/gsl_statistics.h>

using namespace std;

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

		int end = roi.x + roi.width;
		p1.x = end + t;
		p2.x = end - t;

		i1 = segments.at<int32_t>(p1);
		i2 = segments.at<int32_t>(p2);

		if(i1 == i2){
			span.insert(i1);
		}

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

vector<KeyPoint> getAllMatchedKeypoints( map_kp& finalPointClusters){
	vector<KeyPoint> kp;

	for(map<int, vector<KeyPoint> >::iterator it = finalPointClusters.begin(); it != finalPointClusters.end(); ++it){
		kp.insert(kp.end(), it->second.begin(), it->second.end());
	}
	return kp;
}

double countPrint(map_t& roiClusterPoints, map_kp& clusterKeyPoints, vector<int32_t>& cest, int32_t& selectedFeatures, double& lsize){
	double total = 0;
	for (map<int, vector<int>>::iterator it = roiClusterPoints.begin(); it != roiClusterPoints.end(); ++it) {
		if(it->first != 0){
			int32_t n = clusterKeyPoints[it->first].size() / it->second.size();
			cest.push_back(n);
			total += n;
			printf("%d has %lu and total is %lu :: Approx Num of objects: %d\n\n", it->first, it->second.size(),
					clusterKeyPoints[it->first].size(), n);
			selectedFeatures += clusterKeyPoints[it->first].size();

			if (clusterKeyPoints[it->first].size() > lsize) {
				//f.largest = it->first;
				lsize = clusterKeyPoints[it->first].size();
			}
		}
	}
	return total;
}

void generateFinalPointClusters(map_kp& finalPointClusters, map_t& roiClusterPoints, map_kp& clusterKeyPoints){
//framed  generateFinalPointClusters(framed& f, map<int, int> pointsMap){
	//framed f1;
	//map<int, int> pointsMap = splitROIPoints(f, f1);

	for(map<int, vector<int>>::iterator it = roiClusterPoints.begin(); it != roiClusterPoints.end(); ++it) {
		if (it->first != 0 && it->second.size() == 1) {
			int ptIdx = it->second[0];
			finalPointClusters[ptIdx] = clusterKeyPoints[it->first];
		}
	}

	/*for(map<int, vector<int>>::iterator it = f1.roiClusterPoints.begin(); it != f1.roiClusterPoints.end(); ++it) {
		if(it->first != 0 && it->second.size() == 1){
			int nptIdx = it->second[0];
			int optIdx = pointsMap[nptIdx];
			f.finalPointClusters[optIdx] = f1.clusterKeyPoints[it->first];
		}
	}*/

	if(finalPointClusters.size() == 0){

		for(map<int, vector<int>>::iterator it = roiClusterPoints.begin(); it != roiClusterPoints.end(); ++it) {
			if (it->first != 0) {
				int ptIdx = it->second[0];
				finalPointClusters[ptIdx] = clusterKeyPoints[it->first];
			}
		}

		/*for(map<int, vector<int>>::iterator it = f1.roiClusterPoints.begin(); it != f1.roiClusterPoints.end(); ++it) {
			if(it->first != 0){
				int nptIdx = it->second[0];
				int optIdx = pointsMap[nptIdx];
				f.finalPointClusters[optIdx] = f1.clusterKeyPoints[it->first];
			}
		}*/
	}

	//return f1;
}

void mapClusters(vector<int>& labels, map_kp& clusterKeyPoints, map_t& clusterKeypointIdx, vector<KeyPoint>& keypoints){

	// Map cluster labels to the indices of the points
	for(uint i = 0; i < labels.size(); i++){
		int l = labels[i];
		clusterKeyPoints[l].push_back(keypoints[i]);
		clusterKeypointIdx[l].push_back(i);
	}
}

map_t mapSampleFeatureClusters(vector<int>& roiFeatures, vector<int>& labels){

	map_t rcp;
	// get a cluster labels belonging to the sample features and map them the KeyPoint indexes
	for (vector<int>::iterator it = roiFeatures.begin(); it != roiFeatures.end(); ++it) {
		int key = labels[*it];
		rcp[key].push_back(*it);
	}
	return rcp;
}


/**
 *
 */
void generateClusterImages(Mat frame, map_kp& finalPointClusters, map<String, Mat>& keyPointImages, vector<int32_t>& cest, double& total, double& lsize, int32_t& selectedFeatures){

	vector<KeyPoint> kp;
	for(map<int, vector<KeyPoint> >::iterator it = finalPointClusters.begin(); it != finalPointClusters.end(); ++it){
		cest.push_back(it->second.size());
		total += it->second.size();
		printf("Point %d has %lu objects\n", it->first, it->second.size());
		selectedFeatures += it->second.size();

		if(it->second.size() > lsize){
			//f.largest = it->first;
			lsize = it->second.size();
		}

		Mat kimg = drawKeyPoints(frame, it->second, Scalar(0, 0, 255), -1);

		String ss = "img_keypoints-";
		string s = to_string(keyPointImages.size());
		ss += s.c_str();
		keyPointImages[ss] = kimg;
		kp.insert(kp.end(), it->second.begin(), it->second.end());
	}

	Mat mm = drawKeyPoints(frame, kp, Scalar(0, 0, 255), -1);

	String ss = "img_allkps";
	keyPointImages[ss] = mm;
}

void maintaintHistory(vocount& voc, framed& f){
	voc.frameHistory.push_back(f);
	if(voc.frameHistory.size() > 10){
		voc.frameHistory.erase(voc.frameHistory.begin());
	}
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

void getDataset(vocount& vcount, results_t& res, Mat descriptors){
	Mat dataset = descriptors.clone();

	if (!vcount.frameHistory.empty()) {
		for (int j = 1; j < vcount.step; ++j) {
			int ix = vcount.frameHistory.size() - j;
			if (ix >= 0) {
				framed fx = vcount.frameHistory[ix];
				dataset.push_back(fx.descriptors);
			}
		}
	}

	res.ogsize = dataset.rows;
	res.dataset = dataset;
	///res.length = res.dataset.
	//return res.ogsize;
}

/**
 *
 */
double calcDistanceL1(Point2f f1, Point2f f2){
	double diff = f1.x - f2.x;
	double sum = diff * diff;

	diff = f1.y - f2.y;
	sum += diff * diff;

	return sum;
}

/**
 * Find the roi features and at the same time find the central feature.
 */
void findROIFeature(framed& f, selection_t& sel){
	Rect2d r = f.roi;

	Point2f p;

	p.x = (r.x + r.width)/2.0f;
	p.y = (r.y + r.height)/2.0f;
	double distance;

	for(uint i = 0; i < f.keypoints.size(); ++i){
		bool selected = sel.minPts == -1 || (sel.minPts != -1 && std::find(sel.selectedPtsIdx.begin(),	sel.selectedPtsIdx.end(), i) != sel.selectedPtsIdx.end());
		if(f.hasRoi && f.roi.contains(f.keypoints[i].pt) && selected){
			f.roiFeatures.push_back(i);

			// find the center feature index
			if(f.centerFeature == -1){
				f.centerFeature = i;
				distance = calcDistanceL1(p, f.keypoints[i].pt);
			} else {
				double d1 = calcDistanceL1(p, f.keypoints[i].pt);

				if(d1 < distance){
					distance = d1;
					f.centerFeature = i;
				}
			}

			// create the roi descriptor
			if(f.roiDesc.empty()){
				f.roiDesc = f.descriptors.row(i);
			} else{
				f.roiDesc.push_back(f.descriptors.row(i));
			}
		}
	}
}

bool processOptions(vocount& vcount, CommandLineParser& parser, VideoCapture& cap){

	if (parser.has("o")) {
		vcount.destFolder = parser.get<String>("o");
		vcount.print = true;
		printf("Will print to %s\n", vcount.destFolder.c_str());
	}

	if (parser.has("v") || parser.has("video")) {

		vcount.inputPath =
				parser.has("v") ?
						parser.get<String>("v") : parser.get<String>("video");
		cap.open(vcount.inputPath);
	} else {
		printf("You did not provide the video stream to open.");
		return false;
	}

	if (parser.has("w")) {

		String s = parser.get<String>("w");
		vcount.step = atoi(s.c_str());
	} else {
		vcount.step = 1;
	}

	if (parser.has("n")) {
		String s = parser.get<String>("n");
		vcount.rsize = atoi(s.c_str());
	}
	return true;
}

int rectExist(vector<box_structure> structures, Rect& r){

	double maxIntersect = 0.0;
	//Rect maxRec(0, 0, 0, 0);
	int maxIndex = -1;

	for(uint i = 0; i < structures.size(); i++){
		Rect r2 = r & structures[i].box;
		double sect = ((double)r2.area()/r.area()) * 100;
		if(sect > maxIntersect){
			maxIndex = i;
			//maxRec = r2;
			maxIntersect = sect;
		}
	}


	if(maxIntersect > 50.0){
		return maxIndex;
	}

	return -1;
}

void boxStructure(map_kp& finalPointClusters, vector<KeyPoint>& keypoints, Rect2d& roi, vector<box_structure>& boxStructures, map<String, Mat>& keyPointImages){// framed& f){
	box_structure mbs;
	mbs.box = roi;

	for(map<int, vector<KeyPoint>>::iterator it = finalPointClusters.begin(); it != finalPointClusters.end(); ++it){
		vector<KeyPoint> c_points = it->second;
		KeyPoint roi_p = keypoints[it->first];
		mbs.points.push_back(roi_p);

		for(uint j = 0; j < c_points.size(); j++){
			//int pointIndex = c_points[j];

			KeyPoint point = c_points[j];
			if(point.pt != roi_p.pt){ // roi points have their own structure "mbs"
				Point2f pshift;
				pshift.x = point.pt.x - roi_p.pt.x;
				pshift.y = point.pt.y - roi_p.pt.y;

				// shift the roi to get roi for a possible new object
				Rect nr = roi;

				Point pp = pshift;
				nr = nr + pp;
				// check that the rect does not already exist
				int idx = rectExist(boxStructures, nr);
				if(idx == -1){
					box_structure bst;
					bst.box = nr;
					bst.points.push_back(point);
					boxStructures.push_back(bst);
				} else{
					boxStructures[idx].points.push_back(point);
				}
			}
		}
	}

	boxStructures.push_back(mbs);
	printf("boxStructure found %lu objects\n\n", boxStructures.size());
	String ss = "img_bounds";

	Mat img_bounds = keyPointImages["img_allkps"].clone();
	for (uint i = 0; i < boxStructures.size(); i++) {
		box_structure b = boxStructures[i];
		RNG rng(12345);
		Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
				rng.uniform(0, 255));
		rectangle(img_bounds, b.box, value, 2, 8, 0);
	}
	keyPointImages[ss] = img_bounds;
}

map<int, int> addDescriptors(results_t& f, results_t& f1, int cluster, vector<int> roipts){

	vector<int> crois = f.clusterKeypointIdx[cluster];
	map<int, int> newMap;

	for(uint i = 0; i < crois.size(); i++){
		int rp = crois[i];

		/*f1.dataset.push_back(f.descriptors.row(rp));
		f1.keypoints.push_back(f.keypoints[rp]);
		newMap[f1.dataset.rows-1] = rp;

		for(uint j = 0; j < roipts.size(); j++){
			if(roipts[j] == rp){
				f1.roiFeatures.push_back(i);
				if(f1.roiDesc.empty()){
					f1.roiDesc = f.descriptors.row(rp);
				} else{
					f1.roiDesc.push_back(f.descriptors.row(rp));
				}
			}
		}*/
	}

	return newMap;
}

map<int, int> splitROIPoints(framed& f, framed& f1){

	map<int, int> fToF1;
	f1.i = f.i;
	f1.frame = f.frame;
	f1.hasRoi = f.hasRoi;

	bool run = false;
	/*for(map<int, vector<int>>::iterator it = f.roiClusterPoints.begin(); it != f.roiClusterPoints.end(); ++it){
		if (it->second.size() > 1 && it->first != 0) {

			run = true;
			map<int, int> fm = addDescriptors(f, f1, it->first, it->second);
			fToF1.insert(fm.begin(), fm.end());
		}
	}

	cout << "f.roiClusterPoints[0].size() " << f.roiClusterPoints[0].size() << endl;
	*
	 * If there code above did not need reclustering, we check if the noise cluster has
	 * more roi features, in which case we add it to the reclustering data.

	if(!run && f.roiClusterPoints[0].size() > 0 && f1.dataset.rows == 0){

		run = true;
		map<int, int> fm = addDescriptors(f, f1, 0, f.roiClusterPoints[0]);

		fToF1.insert(fm.begin(), fm.end());
	}

	if(run){
		f1.dataset = f1.dataset.clone();
		f1.descriptors = f1.dataset.clone();
		f1.ogsize = f1.dataset.rows;
		f1.hasRoi = f.hasRoi;

		hdbscan<float> sc(_EUCLIDEAN, 4);
		sc.run(f1.dataset.ptr<float>(), f1.dataset.rows, f1.dataset.cols, true);
		f1.labels = sc.getClusterLabels();
		//mapKeyPoints(f1);
		mapClusters(f.labels, f.clusterKeyPoints, f.clusterKeypointIdx, f.keypoints);
		return fToF1;
	}*/

	return map<int, int>();
}

vector<Point2f> reduceDescriptorDimensions(Mat descriptors){
	vector<Point2f> points;

	for(int i = 0; i < descriptors.rows; i++){
		Point2f p(0, 0);
		for(int j = 0; j < descriptors.cols; i++){
			float f = descriptors.at<float>(i, j);

			if(j %2 == 0){
				p.x += f;
			} else{
				p.y += f;
			}
		}
		points.push_back(p);
	}

	return points;
}

map<int, int> getFrameTruth(String truthFolder){
	map<int, int> trueCount;
	DIR*     dir;
	    dirent*  pdir;

	    dir = opendir(truthFolder.c_str());     // open current directory

	    while (pdir = readdir(dir)) {
	        String s = pdir->d_name;
	        if(s != "." && s != ".."){
	            String full = truthFolder + "/";
	            full = full + pdir->d_name;
	            Mat image = imread(full, CV_LOAD_IMAGE_GRAYSCALE);
				threshold(image, image, 200, 255, THRESH_BINARY);
				Mat labels;
				connectedComponents(image, labels, 8, CV_16U);
				double min, max;
				cv::minMaxLoc(labels, &min, &max);

				char* pch = strtok (pdir->d_name," ");
				int fnum = atoi(pch);
				trueCount[fnum] = int(max);
	        }
	    }

	return trueCount;
}


/**
 *
 */
Mat getDistanceDataset(Mat descriptors, Mat roiDesc){
	Mat dset(descriptors.rows, roiDesc.rows, CV_32FC1);

	float* data = dset.ptr<float>(0);

/*
#ifdef USE_OPENMP
#pragma omp for shared(data)
#endif
*/
	for (size_t i = 0; i < descriptors.rows; ++i) {
		Mat row = descriptors.row(i);
		int x = i * 3;

		for (size_t j = 0; j < roiDesc.rows; ++j) {
			Mat d = roiDesc.row(j);
			float distance = norm(row, d);
/*
#ifdef USE_OPENMP
#pragma omp barrier
#endif
*/
			data[x + j] = distance;
		}
	}

	return dset;
}

/**
 *
 */
Mat getDistanceDataset(Mat descriptors, vector<int> roiIdx){
	Mat dset(descriptors.rows, roiIdx.size(), CV_32FC1);

	float* data = dset.ptr<float>(0);

/*
#ifdef USE_OPENMP
#pragma omp for shared(data)
#endif
*/
	for (size_t i = 0; i < descriptors.rows; ++i) {
		Mat row = descriptors.row(i);
		int x = i * 3;

		for (size_t j = 0; j < roiIdx.size(); ++j) {
			Mat d = descriptors.row(roiIdx[j]);
			float distance = norm(row, d);
/*
#ifdef USE_OPENMP
#pragma omp barrier
#endif
*/
			data[x + j] = distance;
		}
	}

	return dset;
}


Mat getColourDataset(Mat f, vector<KeyPoint> pts){
	cout << "getting stdata" << endl;
	Mat m(pts.size(), 3, CV_32FC1);
	float* data = m.ptr<float>(0);
	for(size_t i = 0; i < pts.size(); i++){
		Point2f pt = pts[i].pt;
		Vec3b p = f.at<Vec3b>(pt);
		int idx = i * 3;
		data[idx] = p.val[0];
		data[idx + 1] = p.val[1];
		data[idx + 2] = p.val[2];
	}
	cout << "getting stdata done" << endl;
	return m;
}

map_d getMinMaxDistances(map_t mp, hdbscan<float>& sc, double* core){

	map_d pm;
	double zero = 0.00000000000;

	for(map_t::iterator it = mp.begin(); it != mp.end(); ++it){
		vector<int> idc = it->second;

		for(size_t i = 0; i < idc.size(); i++){

			// min and max core distances
			if(pm[it->first].size() == 0){
				pm[it->first].push_back(core[idc[i]]);
				pm[it->first].push_back(core[idc[i]]);
			} else{
				// min core distance
				if(pm[it->first][0] > core[idc[i]] && (core[idc[i]] < zero || core[idc[i]] > zero)){
					pm[it->first][0] = core[idc[i]];
				}

				//max core distance
				if(pm[it->first][1] < core[idc[i]]){
					pm[it->first][1] = core[idc[i]];
				}
			}

			// Calculating min and max distances
			for(size_t j = i+1; j < idc.size(); j++){
				double d = sc.getDistance(i, j);

				if(pm[it->first].size() == 2){
					pm[it->first].push_back(d);
					pm[it->first].push_back(d);
				} else{
					// min distance
					if(pm[it->first][2] > d && (d < zero || d > zero)){
						pm[it->first][2] = d;
					}

					// max distance
					if(pm[it->first][3] < d){
						pm[it->first][3] = d;
					}
				}
			}
		}
	}

	return pm;
}

map<String, double> getStatistics(map_d distances, double* cr, double* dr){

	map<String, double> stats;
	//double cr[distances.size()];
	//double dr[distances.size()];

	int c = 0;
	for(map_d::iterator it = distances.begin(); it != distances.end(); ++it){

		cr[c] = it->second[1]/it->second[0];

		/*if(it->second[0] == 0.0){
			cr[c] = numeric_limits<double>::max();
		}*/

		/*if(cr[c] != cr[c]){
			printf("**********cr[c] is nan\n");
		}*/

		dr[c] = it->second[3]/it->second[2];
		/*if(it->second[2] == 0.0){
			dr[c] = numeric_limits<double>::max();
		}*/
		/*if(dr[c] != dr[c]){
			printf("**********dr[c] is nan\n");
		}


		printf("cr : [%4f/%4f = %4f] | dr : [%4f/%4f = %4f]\n", it->second[1], it->second[0], cr[c], it->second[3], it->second[2], dr[c]);*/
		c++;
	}
	/*printf("cr = [");
	for(int i = 0; i < distances.size(); i++){
		printf("%.4f, ", cr[i]);
	}
	printf("] \n\ndr = [");
	for(int i = 0; i < distances.size(); i++){
		printf("%.4f, ", dr[i]);
	}
	printf("]\n\n");*/

	// Calculating core distance statistics
	stats["mean_cr"] = gsl_stats_mean(cr, 1, c);
	stats["sd_cr"] = gsl_stats_sd(cr, 1, c);
	stats["variance_cr"] = gsl_stats_variance(cr, 1, c);
	stats["max_cr"] = gsl_stats_max(cr, 1, c);
	stats["kurtosis_cr"] = gsl_stats_kurtosis(cr, 1, c);
	stats["skew_cr"] = gsl_stats_skew(cr, 1, c);

	stats["mean_dr"] = gsl_stats_mean(dr, 1, c);
	stats["sd_dr"] = gsl_stats_sd(dr, 1, c);
	stats["variance_dr"] = gsl_stats_variance(dr, 1, c);
	stats["max_dr"] = gsl_stats_max(dr, 1, c);
	stats["kurtosis_dr"] = gsl_stats_kurtosis(dr, 1, c);
	stats["skew_dr"] = gsl_stats_skew(dr, 1, c);
	stats["count"] = c;

	/*printf("stats[\"mean_cr\"] = %f\n", stats["mean_cr"]);
	printf("stats[\"sd_cr\"] = %f\n", stats["sd_cr"]);
	printf("stats[\"variance_cr\"] = %f\n", stats["variance_cr"]);
	printf("stats[\"max_cr\"] = %f\n", stats["max_cr"]);
	printf("stats[\"kurtosis_cr\"] = %f\n", stats["kurtosis_cr"]);
	printf("stats[\"skew_cr\"] = %f\n", stats["skew_cr"]);

	// Calculating distance statistics
	printf("\n\n******************\nstats[\"mean_dr\"] = %f\n", stats["mean_dr"]);
	printf("stats[\"sd_dr\"] = %f\n", stats["sd_dr"]);
	printf("stats[\"variance_dr\"] = %f\n", stats["variance_dr"]);
	printf("stats[\"max_dr\"] = %f\n", stats["max_dr"]);
	printf("stats[\"kurtosis_dr\"] = %f\n", stats["kurtosis_dr"]);
	printf("stats[\"skew_dr\"] = %f\n", stats["skew_dr"]);

	printf("\nstats[\"count\"] = %f\n", stats["count"]);*/

	//int validity = analyseStats(stats);

	//printf("valididty -----> %d\n", validity);

	return stats;
}

int analyseStats(map<String, double> stats){
	 int validity = -1;
	if((stats["skew_dr"] > 0.0 || stats["skew_dr"] != stats["skew_dr"]) && (stats["kurtosis_dr"] > 0.0 || stats["kurtosis_dr"] != stats["kurtosis_dr"])){
		validity = 2;
	} else if(stats["skew_dr"] < 0.0 && stats["kurtosis_dr"] > 0.0){
		validity = 1;
	} else if(stats["skew_dr"] > 0.0 && stats["kurtosis_dr"] < 0.0){
		validity = 0;
	} else{
		validity = -1;
	}

	if((stats["skew_cr"] > 0.0 || stats["skew_cr"] != stats["skew_cr"]) && (stats["kurtosis_cr"] > 0.0 || stats["kurtosis_cr"] != stats["kurtosis_cr"])){
		validity += 2;
	} else if(stats["skew_cr"] < 0.0 && stats["kurtosis_cr"] > 0.0){
		validity += 1;
	} else if(stats["skew_cr"] > 0.0 && stats["kurtosis_cr"] < 0.0){
		validity += 0;
	} else{
		validity += -1;
	}

	return validity;
}

Mat getSelectedKeypointsDescriptors(Mat desc, vector<int> indices){
	Mat m;
	for(size_t i = 0; i < indices.size(); i++){
		if(m.empty()){
			m = desc.row(indices[i]);

		} else{
			m.push_back(desc.row(indices[i]));
		}
	}

	return m;
}

/**
 *
 */
selection_t detectColourSelectionMinPts(Mat frame, Mat descriptors, vector<KeyPoint> keypoints){
	int mpts;
	printf("Detecting minPts value for colour clustering.\n");
	map<int, map<String, double>> stats;
	Mat dataset = getColourDataset(frame, keypoints);
	size_t size = 0;
	map<int, int> choices;
	int chosenCount = 1, currentCount = 1;
	map_t clusterKeypointIdxMap;
	map_kp clusterKeyPointsMap;
	selection_t colourSelection;
	colourSelection.minPts = 2;

	for(int i = 3; i < 30; i++){

		printf("\n\n >>>>>>>>>>>>>>>>>>>>>>>>> Clustering for minPts = %d\n", i);
		map_kp clusterKeyPoints;
		map_t clusterKeypointIdx;

		hdbscan<float> scanc(_EUCLIDEAN, i);
		scanc.run(dataset.ptr<float>(), dataset.rows, dataset.cols, true);
		vector<int> labels = scanc.getClusterLabels();
		set<int> lsetkps(labels.begin(), labels.end());
		double* corecl = scanc.getCoreDistances();
		mapClusters(labels, clusterKeyPoints, clusterKeypointIdx, keypoints);
		map_d disMapCl = getMinMaxDistances(clusterKeypointIdx, scanc, corecl);
		double cr[disMapCl.size()];
		double dr[disMapCl.size()];
		stats[i] = getStatistics(disMapCl, cr, dr);
		//int v = analyseStats(stats[i]);

		if(disMapCl.size() != size){
			size = disMapCl.size();
			mpts = i;
			currentCount = 1;
		} else{
			currentCount++;
			if(currentCount > chosenCount){
				chosenCount = currentCount;
				colourSelection.minPts = mpts;
				clusterKeypointIdxMap = clusterKeypointIdx;
				clusterKeyPointsMap = clusterKeyPoints;
			}
		}

	}
	printf(">>>>>>>> VALID CHOICE OF minPts IS %d <<<<<<<<<\n", colourSelection.minPts);
	//printStatistics(stats, "./");
	for(map_t::iterator it = clusterKeypointIdxMap.begin(); it != clusterKeypointIdxMap.end(); ++it){

		Mat m = drawKeyPoints(frame, clusterKeyPointsMap[it->first], Scalar(0, 0, 255), -1);
		display("choose", m);
		// Listen for a key pressed
		char c = ' ';
		while(true){
			if (c == 'a') {
				Mat xx = getSelectedKeypointsDescriptors(descriptors, it->second);
				colourSelection.selectedPts.insert(colourSelection.selectedPts.end(), clusterKeyPointsMap[it->first].begin(), clusterKeyPointsMap[it->first].end());
				colourSelection.selectedPtsIdx.insert(colourSelection.selectedPtsIdx.end(), clusterKeypointIdxMap[it->first].begin(), clusterKeypointIdxMap[it->first].end());
				if(colourSelection.selectedDesc.empty()){
					colourSelection.selectedDesc = xx.clone();
				} else{
					colourSelection.selectedDesc.push_back(xx);
				}
				break;
			} else if (c == 'q'){
				break;
			}
			c = (char) waitKey(20);
		}
		destroyWindow("choose");
	}
	return colourSelection;
}


Mat getPointDataset(vector<KeyPoint> keypoints){
	Mat m(keypoints.size(), 2, CV_32FC1);
	float *data = m.ptr<float>(0);
	for(size_t i = 0; i < keypoints.size(); i++){
		int idx = i *2;
		data[idx] = keypoints[i].pt.x;
		data[idx+1] = keypoints[i].pt.y;
	}
	return m;
}
