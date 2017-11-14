/*
 * process_frame.cpp
 *
 *  Created on: 3 May 2017
 *      Author: ojmakh
 */

#include "vocount/process_frame.hpp"
#include <fstream>
#include <opencv/cv.hpp>
#include <opencv2/imgproc.hpp>
#include <dirent.h>
#include <gsl/gsl_statistics.h>

using namespace std;

void display(char const* screen, const InputArray& m) {
	if (!m.empty()) {
		namedWindow(screen, WINDOW_AUTOSIZE);
		imshow(screen, m);
	}
}

results_t* splitROICluster(IntArrayList* list, Mat* dataset, vector<KeyPoint>* keypoints, vector<int32_t>& oldIdx){
	
	int32_t* l1d = (int32_t *)list->data;
	Mat dset(0, dataset->cols, CV_32FC1);
	vector<KeyPoint> kp;
	for(int i = 0; i < list->size; i++){
		int idxx = l1d[i];
		dset.push_back(dataset->row(idxx));
		kp.push_back(keypoints->at(idxx));
		oldIdx.push_back(idxx);
	}
	
	results_t* res = initResult_t(dset, kp);
	res->minPts = 3;
	hdbscan scan(res->minPts, DATATYPE_FLOAT);
	scan.run(res->dataset->ptr<float>(), res->dataset->rows, res->dataset->cols, TRUE);
	res->labels->insert(res->labels->begin(), scan.clusterLabels, scan.clusterLabels+scan.numPoints);
	res->clusterMap = hdbscan_create_cluster_table(scan.clusterLabels, 0, scan.numPoints);
	
	res->distancesMap = hdbscan_get_min_max_distances(&scan, res->clusterMap);
	res->stats = hdbscan_calculate_stats(res->distancesMap);
	res->validity = hdbscan_analyse_stats(res->stats);
	
	//hdbscan_print_cluster_table(res->clusterMap);
	return res;
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

double countPrint(IntIntListMap* roiClusterPoints, map_kp* clusterKeyPoints, vector<int32_t>* cest, int32_t& selectedFeatures, double& lsize){
	double total = 0;
			
	GHashTableIter iter;
	gpointer key;
	gpointer value;
	g_hash_table_iter_init (&iter, roiClusterPoints);

	while (g_hash_table_iter_next (&iter, &key, &value)){
		
		int32_t* kk = (int32_t *)key;
		IntArrayList* list = (IntArrayList *)value;
		if(*kk != 0){
			int32_t n = (*clusterKeyPoints)[*kk].size() / list->size;
			cest->push_back(n);
			total += n;
			printf("%d has %d and total is %lu :: Approx Num of objects: %d\n\n", *kk, list->size,
					(*clusterKeyPoints)[*kk].size(), n);
			selectedFeatures += (*clusterKeyPoints)[*kk].size();

			if ((*clusterKeyPoints)[*kk].size() > lsize) {
				lsize = (*clusterKeyPoints)[*kk].size();
			}
		}		
	}
	return total;
}

void generateFinalPointClusters(vector<vector<int32_t>>& roiFeatures, IntIntListMap* clusterMap, IntIntListMap* roiClusterPoints, map_kp* finalPointClusters, vector<int32_t>* labels, vector<KeyPoint>* keypoints){
	set<int32_t> st;
	for(uint i = 0; i < roiFeatures.size(); i++){
		for (vector<int32_t>::iterator it = roiFeatures[i].begin(); it != roiFeatures[i].end(); ++it) {
			int* key;
			int k = labels->at(*it);
			//res->objectClusters->insert(k);
			key = &k;
			IntArrayList* list = (IntArrayList *)g_hash_table_lookup(roiClusterPoints, key);
			
			if(list == NULL){
				key = (int *)malloc(sizeof(int));
				*key = k;
				list = int_array_list_init_size(roiFeatures[i].size());
				g_hash_table_insert(roiClusterPoints, key, list);
				st.insert(k);
			}
					
			int_array_list_append(list, *it);
			
		}
	}
	
	for (set<int32_t>::iterator it = st.begin(); it != st.end(); ++it){
		int32_t key = *it;
		if (key != 0) {
			IntArrayList* l1 = (IntArrayList*)g_hash_table_lookup(clusterMap, &key);			
			getListKeypoints(*keypoints, l1, (*(finalPointClusters))[key]);
		}		
	}
}

int rectExist(vector<box_structure>& structures, Rect& r){

	double maxIntersect = 0.0;
	int maxIndex = -1;

	for(uint i = 0; i < structures.size(); i++){
		Rect r2 = r & structures[i].box;
		double sect = ((double)r2.area()/r.area()) * 100;
		if(sect > maxIntersect){
			maxIndex = i;
			maxIntersect = sect;
		}
	}


	if(maxIntersect > 50.0){
		return maxIndex;
	}

	return -1;
}

void addToBoxStructure(vector<box_structure>* boxStructures, vector<KeyPoint> c_points, KeyPoint first_p, box_structure& mbs, Mat& frame){
	
	for(uint j = 0; j < c_points.size(); j++){
			
		KeyPoint point = c_points[j];
		if(point.pt != first_p.pt){ // roi points have their own structure "mbs"
			Point2f pshift;
			pshift.x = point.pt.x - first_p.pt.x;
			pshift.y = point.pt.y - first_p.pt.y;

			// shift the roi to get roi for a possible new object
			Rect n_rect = mbs.box;

			Point pp = pshift;
			n_rect = n_rect + pp;
			
			if(n_rect.x < 0 || n_rect.y < 0 || (n_rect.x + n_rect.width) >= frame.cols || (n_rect.y + n_rect.height) >= frame.rows){
				cout << "Skipping " << n_rect << endl;
				continue;
			}
			
			// check that the rect does not already exist
			int idx = rectExist(*boxStructures, n_rect);
			if(idx == -1){
				box_structure bst;
				bst.box = n_rect;
				bst.points.push_back(point);
								
				//cout << mbs.box << " : " << n_rect;
				
				if(n_rect.x < 0){
					n_rect.width += n_rect.x;
					n_rect.x = 0;
				}
				
				if(n_rect.y < 0){
					n_rect.height += n_rect.y;
					n_rect.y = 0;
				}
				
				if((n_rect.x + n_rect.width) >= frame.cols){
					n_rect.width = frame.cols - n_rect.x;
				}
				
				if((n_rect.y + n_rect.height) >= frame.rows){
					n_rect.height = frame.rows - n_rect.y;
				}
				
				double area1 = bst.box.width * bst.box.width; 
				double area2 = n_rect.width * n_rect.width;
				double ratio = area2/area1;
				//if(ratio < 0.5){
					//cout << "Ratio is " << ratio<< " Skipping " << n_rect << endl;
					//continue;
				//}
				
				//cout << " (" << n_rect << ") compare ";
								
				bst.img_ = frame(n_rect);
				calculateHistogram(bst);
				bst.histCompare = compareHist(mbs.hist, bst.hist, CV_COMP_CORREL);
				
				Mat g1, g2;
				cvtColor(mbs.img_, g1, COLOR_RGB2GRAY);
				cvtColor(bst.img_, g2, COLOR_RGB2GRAY);
				bst.momentsCompare = matchShapes(g1, g2, CONTOURS_MATCH_I3, 0);
				cout << " (" << n_rect << ") compare " << bst.histCompare << " moments compare " << bst.momentsCompare << endl;
				if(bst.momentsCompare > 0.05){
					cout << "Skipping for low similarity" << endl;
					continue;
				}
				boxStructures->push_back(bst);
			} else{
				(*boxStructures)[idx].points.push_back(point);
			}
		}
	}
}

/**
 * Find clusters that have points inside one of the bounding boxes
 * 
 */ 
void extendBoxClusters(Mat& frame, vector<box_structure>* boxStructures, vector<KeyPoint>& keypoints, map_kp* finalPointClusters, IntIntListMap* clusterMap, IntDoubleListMap* distanceMap){
	
}


/**
 *
 */
void generateClusterImages(Mat frame, results_t* res){

	vector<KeyPoint> kp;
	for(map<int, vector<KeyPoint> >::iterator it = res->finalPointClusters->begin(); it != res->finalPointClusters->end(); ++it){
		res->cest->push_back(it->second.size());
		res->total += it->second.size();
		res->selectedFeatures += it->second.size();

		if(it->second.size() > res->lsize){
			res->lsize = it->second.size();
		}

		Mat kimg = drawKeyPoints(frame, it->second, Scalar(0, 0, 255), -1);

		String ss = "img_keypoints-";
		string s = to_string(it->first);
		ss += s.c_str();
		(*(res->keyPointImages))[ss] = kimg;
		kp.insert(kp.end(), it->second.begin(), it->second.end());
	}

	Mat mm = drawKeyPoints(frame, kp, Scalar(0, 0, 255), -1);

	String ss = "img_allkps";
	(*(res->keyPointImages))[ss] = mm;
}

void maintaintHistory(vocount& voc, framed& f){
	voc.frameHistory.push_back(f);
	if(voc.frameHistory.size() > 10){
		framed& f1 = voc.frameHistory.front();
		
		for(map<String, results_t*>::iterator it = f1.results.begin(); it != f1.results.end(); ++it){
			printf("Cleaning results %s\n", it->first.c_str());
			cleanResult(it->second);
		}
		
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

Mat getDescriptorDataset(vector<framed>& frameHistory, int step, Mat descriptors){
	Mat dataset = descriptors.clone();

	if (!frameHistory.empty()) {
		for (int j = 1; j < step; ++j) {
			int ix = frameHistory.size() - j;
			if (ix >= 0) {
				framed fx = frameHistory[ix];
				dataset.push_back(fx.descriptors);
			}
		}
	}

	return dataset;
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
int32_t findROIFeature(vector<KeyPoint>& keypoints, Mat& descriptors, vector<Rect2d>& rois, vector<vector<int32_t>>& roiFeatures, vector<Mat>& roiDesc){
	printf("f.rois has %lu\n", rois.size());
	roiFeatures.reserve(rois.size());
	roiDesc.reserve(rois.size());
	
	for(uint i = 0; i < rois.size(); i++){
		roiFeatures.push_back(vector<int32_t>());
		roiDesc.push_back(Mat());
	}
	
	Rect2d r = rois[0];

	Point2f p;

	p.x = (r.x + r.width)/2.0f;
	p.y = (r.y + r.height)/2.0f;
	double distance;
	int32_t centerFeature = -1;
	for(uint i = 0; i < keypoints.size(); ++i){
		uint j = 0;
		
		while(j < rois.size()){		
		
			if(rois[j].contains(keypoints[i].pt)){
				roiFeatures[j].push_back(i);

				// find the center feature index
				if(centerFeature == -1){
					centerFeature = i;
					distance = calcDistanceL1(p, keypoints[i].pt);
				} else {
					double d1 = calcDistanceL1(p, keypoints[i].pt);

					if(d1 < distance){
						distance = d1;
						centerFeature = i;
					}
				}

				// create the roi descriptor
				roiDesc[j].push_back(descriptors.row(i));
				
			}
			j++;
		}
	}
	//printf("roiDesc had %d rows\n", roiDesc.rows);
	return centerFeature;
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

/**
 * Creates box_structure objects from final point clusters
 * 
 */ 
void boxStructure(map_kp* finalPointClusters, vector<KeyPoint>& keypoints, vector<Rect2d>& rois, vector<box_structure>* boxStructures, Mat& frame){
	box_structure mbs;
	mbs.box = rois[0];
	mbs.img_ = frame(mbs.box);
	calculateHistogram(mbs);
	mbs.histCompare = compareHist(mbs.hist, mbs.hist, CV_COMP_CORREL);
	Mat g1;
	cvtColor(mbs.img_, g1, COLOR_RGB2GRAY);
	mbs.momentsCompare = matchShapes(g1, g1, CONTOURS_MATCH_I3, 0);
	boxStructures->push_back(mbs);
	cout << "First box : " << boxStructures->at(0).box << " - " << boxStructures->at(0).momentsCompare << endl;

	for(map_kp::iterator it = finalPointClusters->begin(); it != finalPointClusters->end(); ++it){
		vector<KeyPoint>& kps = it->second;
		KeyPoint kp;
		// here we are looking for the point that is inside the roi for use as a point
		// of reference with the other cluster points
		for(vector<KeyPoint>::iterator itr = kps.begin(); itr != kps.end(); ++itr){
			if(rois[0].contains(itr->pt)){
				kp = *itr;
				break;
			}
		}
		
		mbs.points.push_back(kp);
		addToBoxStructure(boxStructures, it->second, kp, mbs, frame);
	}
	
	/*for(vector<box_structure>::iterator it = boxStructures->begin(); it != boxStructures->end(); ++it){
		cout << "Box : " << boxStructures->at(0).box << " - " << boxStructures->at(0).hist << endl;
	}*/

}

/**
 * Given a vector of box structures, the function draws the rectangles around the identified object locations
 * 
 */ 
void createBoxStructureImages(vector<box_structure>* boxStructures, map<String, Mat>* keyPointImages){
	
	printf("boxStructure found %lu objects\n\n", boxStructures->size());
	String ss = "img_bounds";

	Mat img_bounds = (*keyPointImages)["img_allkps"].clone();
	for (size_t i = 0; i < boxStructures->size(); i++) {
		box_structure b = (*boxStructures)[i];
		RNG rng(12345);
		Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
				rng.uniform(0, 255));
		rectangle(img_bounds, b.box, value, 2, 8, 0);
		//Point center = (b.box.br() + b.box.tl())/2;
		//circle(img_bounds, center, 4, Scalar(255, 255, 255), CV_FILLED, 10, 0);
	}
	(*keyPointImages)[ss] = img_bounds;
}

void getFrameTruth(String truthFolder, vector<int32_t>& truth){
	//map<int, int> trueCount;
	DIR*     dir;
	dirent*  pdir;
	dir = opendir(truthFolder.c_str());     // open current directory
	pdir = readdir(dir);
	while (pdir) {
	    
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
			int fnum = atoi(pch)-1;
			truth.insert(truth.begin() + fnum, int(max));
		}
		pdir = readdir(dir);
	}
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
	for (int i = 0; i < descriptors.rows; ++i) {
		Mat row = descriptors.row(i);
		int x = i * 3;

		for (int j = 0; j < roiDesc.rows; ++j) {
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
	for (int i = 0; i < descriptors.rows; ++i) {
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
	return m;
}

void getSelectedKeypointsDescriptors(Mat& desc, IntArrayList* indices, Mat& out){
	//Mat m;
	//printf("desc has %d rows \n", desc.rows);
	int32_t *dt = (int32_t *)indices->data;
	for(int i = 0; i < indices->size; i++){
		/*if(m.empty()){
			m = desc.row(dt[i]);

		} else{*/
		//printf("Adding row %d\n", dt[i]);
		out.push_back(desc.row(dt[i]));
		//}
	}

	//return m;
}

void getKeypointMap(IntIntListMap* listMap, vector<KeyPoint>* keypoints, map_kp& mp){
	
	GHashTableIter iter;
	gpointer key;
	gpointer value;
	g_hash_table_iter_init (&iter, listMap);

	while (g_hash_table_iter_next (&iter, &key, &value)){
		IntArrayList* clusterLabels = (IntArrayList*)value;
		//int32_t* idxList = (int32_t* )clusterLabels->data;
		int32_t* k = (int32_t *) key;
		getListKeypoints(*keypoints, clusterLabels, mp[*k]);
		/*for(int i = 0; i < clusterLabels->size; i++){
			int idx = idxList[i];
			mp[k].push_back((*keypoints)[idx]);
		}*/
	}
}

void getListKeypoints(vector<KeyPoint>& keypoints, IntArrayList* list, vector<KeyPoint>& out){
	int32_t* dt = (int32_t *)list->data;
	for(int i = 0; i < list->size; i++){
		int32_t idx = dt[i];
		out.push_back(keypoints[idx]);
	}
}

/**
 *
 */
selection_t detectColourSelectionMinPts(Mat& frame, Mat& descriptors, vector<KeyPoint>& keypoints){
	int mpts;
	printf("Detecting minPts value for colour clustering.\n");
	Mat dataset = getColourDataset(frame, keypoints);
	size_t size = 0;
	map<int, int> choices;
	int chosenCount = 1, currentCount = 1;
	IntIntListMap* clusterKeypointIdxMap = NULL;
	map_kp clusterKeyPointsMap;
	selection_t colourSelection;
	colourSelection.minPts = 2;

	for(int i = 3; i < 30; i++){

		printf("\n\n >>>>>>>>>>>> Clustering for minPts = %d\n", i);
		hdbscan scan(i, DATATYPE_FLOAT);
		scan.run(dataset.ptr<float>(), dataset.rows, dataset.cols, TRUE);		
		set<int> lsetkps(scan.clusterLabels, scan.clusterLabels + scan.numPoints);	
			
		IntIntListMap* clusterMap = hdbscan_create_cluster_table(scan.clusterLabels, 0, scan.numPoints);		
		IntDoubleListMap* distancesMap = hdbscan_get_min_max_distances(&scan, clusterMap);
		
		if(g_hash_table_size(distancesMap) != size){
			size = g_hash_table_size(distancesMap);
			mpts = i;
			currentCount = 1;
			hdbscan_destroy_cluster_table(clusterMap);
		} else{
			currentCount++;
			if(currentCount > chosenCount){
				chosenCount = currentCount;
				colourSelection.minPts = mpts;
				
				if(clusterKeypointIdxMap != NULL){
					hdbscan_destroy_cluster_table(clusterKeypointIdxMap);
				}
				
				clusterKeypointIdxMap = clusterMap;
			} else{
				hdbscan_destroy_cluster_table(clusterMap);
			}
		}
		hdbscan_destroy_distance_map_table(distancesMap);

	}
	
	colourSelection.clusterKeypointIdx = clusterKeypointIdxMap;
	printf(">>>>>>>> VALID CHOICE OF minPts IS %d <<<<<<<<<\n", colourSelection.minPts);
	GHashTableIter iter;
	gpointer key;
	gpointer value;
	g_hash_table_iter_init (&iter, clusterKeypointIdxMap);

	while (g_hash_table_iter_next (&iter, &key, &value)){
		IntArrayList* list = (IntArrayList*)value;
		int32_t* k = (int32_t *)key;
		
		if(*k != 0){
			vector<KeyPoint> kps;
			getListKeypoints(keypoints, list, kps);
			Mat m = drawKeyPoints(frame, kps, Scalar(0, 0, 255), -1);
			display("choose", m);
			
			// Listen for a key pressed
			char c = ' ';
			while(true){
				if (c == 'a') {
					Mat xx ;
					getSelectedKeypointsDescriptors(descriptors, list, xx);
					colourSelection.selectedClusters.insert(*k);
					colourSelection.selectedKeypoints.insert(colourSelection.selectedKeypoints.end(), kps.begin(), kps.end());
					if(colourSelection.selectedDesc.empty()){
						colourSelection.selectedDesc = xx.clone();
					} else{
						colourSelection.selectedDesc.push_back(xx);
					}
					printf("%%%%%%%%%%%%%% added cluster %d of size %d\n", *k, list->size);
					break;
				} else if (c == 'q'){
					break;
				}
				c = (char) waitKey(20);
			}
			destroyWindow("choose");
		}
	}
	
	return colourSelection;
}


Mat getImageSpaceDataset(vector<KeyPoint> keypoints){
	Mat m(keypoints.size(), 2, CV_32FC1);
	float *data = m.ptr<float>(0);
	for(size_t i = 0; i < keypoints.size(); i++){
		int idx = i *2;
		data[idx] = keypoints[i].pt.x;
		data[idx+1] = keypoints[i].pt.y;
	}
	return m;
}

results_t* initResult_t(Mat& dataset, vector<KeyPoint>& keypoints){
	results_t* res = (results_t*) malloc(sizeof(results_t));

	res->dataset = new Mat(dataset);
	res->keypoints = new vector<KeyPoint>(keypoints.begin(), keypoints.end());
	res->finalPointClusters = new map_kp();
	res->odata = new map<String, int32_t>();
	res->labels = new vector<int32_t>();
	res->boxStructures = new vector<box_structure>();
	res->cest = new vector<int32_t>();
	res->keyPointImages = new map<String, Mat>();
	res->objectClusters = new set<int32_t>();
		
    res->clusterMap = NULL;		 								/// maps labels to the keypoint indices
    res->roiClusterPoints = g_hash_table_new(g_int_hash, g_int_equal);								/// cluster labels for the region of interest mapped to the roi points in the cluster
    res->stats = NULL;											/// Statistical values for the clusters
    res->distancesMap = NULL;									/// Min and Max distance table for each cluster
    
	res->lsize = 0;
	res->total = 0;
	res->selectedFeatures = 0;
	res->ogsize = 0;
	res->validity = -1;
	res->minPts = 3;
	
	return res;
}

results_t* do_cluster(results_t* res, Mat& dataset, vector<KeyPoint>& keypoints, int step, int f_minPts, bool analyse){
	
	//results_t res;
	if(res == NULL){
		res = initResult_t(dataset, keypoints);
	}
	
	res->minPts = step * f_minPts;
	
	int i = 0;
	
	while(res->validity <= 2 && i < 5){
		res->minPts = (f_minPts + i) * step;
		printf("Testing minPts = %d\n", res->minPts);
		if(res->clusterMap != NULL){
			hdbscan_destroy_cluster_table(res->clusterMap);
		}
		
		if(res->stats != NULL){
			hdbscan_destroy_stats_map(res->stats);
		}
		
		if(res->distancesMap != NULL){
			hdbscan_destroy_distance_map_table(res->distancesMap);
		}
		
		if(!(res->labels->empty())){
			res->labels->clear();
		}
				
		hdbscan scan(res->minPts, DATATYPE_FLOAT);
		scan.run(res->dataset->ptr<float>(), res->dataset->rows, res->dataset->cols, TRUE);
		res->labels->insert(res->labels->begin(), scan.clusterLabels, scan.clusterLabels+scan.numPoints);
		res->clusterMap = hdbscan_create_cluster_table(scan.clusterLabels, 0, scan.numPoints);
		
		if(analyse){
			res->distancesMap = hdbscan_get_min_max_distances(&scan, res->clusterMap);
			res->stats = hdbscan_calculate_stats(res->distancesMap);
			res->validity = hdbscan_analyse_stats(res->stats);
		}
		
		i++;
	}
	res->ogsize = keypoints.size();

	printf("------- Selected max clustering size = %d and cluster table has %d\n", res->minPts, g_hash_table_size(res->clusterMap));
	
	return res;
}

void cleanResult(results_t* res){
	if(res != NULL){
		if(res->clusterMap != NULL){
			hdbscan_destroy_cluster_table(res->clusterMap);
			res->clusterMap = NULL;
		}
		
		if(res->stats != NULL){
			hdbscan_destroy_stats_map(res->stats);
			res->stats = NULL;
		}
		
		if(res->distancesMap != NULL){
			hdbscan_destroy_distance_map_table(res->distancesMap);
			res->distancesMap = NULL;
		}
		
		/**
		 * Here we are using the hdbscan_destroy_cluster_table from the hdbscan.c
		 * because roiClusterPoints and clusterTable are basically the same structure
		 * being IntIntListMap datatype.
		 */ 
		if(res->roiClusterPoints != NULL){		
			hdbscan_destroy_cluster_table(res->roiClusterPoints);
			res->roiClusterPoints = NULL;
		}
		
		delete res->dataset;
		delete res->keypoints;
		delete res->finalPointClusters;
		delete res->odata;
		delete res->labels;
		delete res->boxStructures;
		delete res->cest;
		delete res->keyPointImages;
		delete res->objectClusters;
		free(res);
	}
}

void calculateHistogram(box_structure& bst){	
	cvtColor(bst.img_, bst.hsv, COLOR_BGR2HSV );
	
	/// Using 50 bins for hue and 60 for saturation
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };

    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };

    const float* ranges[] = { h_ranges, s_ranges };

    // Use the o-th and 1-st channels
    int channels[] = { 0, 1 };
    /// Calculate the histograms for the HSV images
    //calcHist(const Mat* images, int nimages, const int* channels, InputArray mask, OutputArray hist, int dims, const int* histSize, const float** ranges, bool uniform = true, bool accumulate = false);
    //calcHist(cv::Mat*, int, int [2], cv::Mat, cv::Mat [3], int, int [2], const float* [2], bool, bool)
    calcHist( &bst.hsv, 1, channels, Mat(), bst.hist, 2, histSize, ranges, true, false );
    normalize( bst.hist, bst.hist, 0, 1, NORM_MINMAX, -1, Mat() );
}
