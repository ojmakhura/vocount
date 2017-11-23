/*
 * process_frame.cpp
 *
 *  Created on: 3 May 2017
 *      Author: ojmakh
 */

#include "vocount/process_frame.hpp"
#include <opencv2/tracking.hpp>
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

void splitROICluster(IntArrayList* roiList, IntArrayList* clusterList, Mat& dataset, vector<vector<KeyPoint>>& splitKeypoints, vector<KeyPoint>& keypoints){
	
	for(int i = 0; i < roiList->size; i++){
		splitKeypoints.push_back(vector<KeyPoint>());
	}
	
	for(int i = 0; i < clusterList->size; i++){
		Mat m1 = dataset.row(i);
		int* rdata = (int *) roiList->data;
		double minNorm = norm (dataset.row(i), dataset.row(rdata[0]), NORM_L1, noArray());		
		int minIdx = 0;
		
		for(int j = 1; j < roiList->size; j++){
			double n1 = norm (dataset.row(i), dataset.row(rdata[j]), NORM_L1, noArray());		
			if(n1 < minNorm){
				minNorm = n1;
				minIdx = j;
			}
		}
		
		splitKeypoints[minIdx].push_back(keypoints[i]);
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

void trimRect(Rect& r, int rows, int cols){
	if(r.x < 0){
		r.width += r.x;
		r.x = 0;
	}
				
	if(r.y < 0){
		r.height += r.y;
		r.y = 0;
	}
				
	if((r.x + r.width) >= cols){
		r.width = cols - r.x;
	}
				
	if((r.y + r.height) >= rows){
		r.height = rows - r.y;
	}
}

bool stabiliseRect(Mat& frame, const Rect& templ_r, Rect& proposed){
	Mat result;
	
	Rect new_r = proposed;
	int half_h = new_r.height/2;
	int half_w = new_r.width/2;
	new_r.x -= half_w/2;
	new_r.y -= half_h/2;
	new_r.width += half_w; //new_r.width;
	new_r.height += half_h; //new_r.height;
	
	trimRect(new_r, frame.rows, frame.cols);
	//cout << "Trimmed new_r  = " << new_r << endl;
	if(new_r.height < 1 || new_r.width < 1){
		return false;
	}
	
	Mat img = frame(new_r);
	Mat templ = frame(templ_r);
	
	if(img.rows < templ.rows && img.cols < templ.cols){
		return false;
	}
	
	int result_cols =  img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
	
	//printf("result dims (%d, %d)\n", result_rows, result_cols);
	
	if(result_rows < 2 || result_cols < 2){
		return false;
	}

	result.create( result_rows, result_cols, CV_32FC1 );
	matchTemplate( img, templ, result, TM_SQDIFF);
	normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;
	minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
	matchLoc = minLoc;
	//cout << "matchLoc is " << matchLoc << endl;
	proposed.x = matchLoc.x + new_r.x;
	proposed.y = matchLoc.y + new_r.y;
	//cout << "Proposed is now " << proposed << endl;
	//trimRect(proposed, frame.rows, frame.cols);
	
	return true;
}

bool stabiliseRectByMoments(Mat& frame, const Rect& templ_r, Rect& proposed){
	//printf("stabiliseRectByMoments(Mat& frame, const Rect& templ_r, Rect& proposed)\n");
	Mat gray;
	cvtColor(frame, gray, COLOR_RGB2GRAY);
	Rect center = proposed;
	//bool centerIsMin = false;
	
	Mat templImg = gray(templ_r);
	int min = -1;
	double minMom;
	
	do {
		vector<Rect> rects;
		trimRect(center, frame.rows, frame.cols);
		Mat centerImg = gray(center);	
		double momCompare;
		if(min == -1){
			momCompare = matchShapes(templImg, centerImg, CONTOURS_MATCH_I3, 0);
			minMom = momCompare;
		} else{
			min = -1;
		}
		
		//cout << "-1 : " << momCompare << " ";
		
		Rect top = center;
		top.y -= 1;
		trimRect(top, frame.rows, frame.cols);
		rects.push_back(top);
				
		Rect bottom = center;
		bottom.y += 1;
		trimRect(bottom, frame.rows, frame.cols);
		rects.push_back(bottom);
				
		Rect right = center;
		right.x += 1;
		trimRect(right, frame.rows, frame.cols);
		rects.push_back(right);
		
		Rect left = center;
		left.x -= 1;
		trimRect(left, frame.rows, frame.cols);
		rects.push_back(left);
		
		Rect topLeft = center;
		topLeft.y -= 1;
		topLeft.x -= 1;
		trimRect(topLeft, frame.rows, frame.cols);
		rects.push_back(topLeft);
				
		Rect bottomLeft = center;
		topLeft.y += 1;
		topLeft.x -= 1;
		trimRect(bottomLeft, frame.rows, frame.cols);
		rects.push_back(bottomLeft);
		
		Rect topRight = center;
		topLeft.y -= 1;
		topLeft.x += 1;
		trimRect(topRight, frame.rows, frame.cols);
		rects.push_back(topRight);
		
		Rect bottomRight = center;
		topLeft.y += 1;
		topLeft.x += 1;
		trimRect(bottomRight, frame.rows, frame.cols);
		rects.push_back(bottomRight);
		
		for(uint i = 0; i < rects.size(); i++){			
			if(rects[i].height < 1 || rects[i].width < 1){
				continue;
			}
			Mat m = gray(rects[i]);
			momCompare = matchShapes(templImg, m, CONTOURS_MATCH_I3, 0);
			//cout <<  i << " : " << momCompare << " ";
			if(momCompare < minMom){
				min = i;
				minMom = momCompare;
			}
		}
		
		//cout << endl;
		
		if(min != -1){
			center = rects[min];
		}
		
	} while(min != -1);
	
	proposed.x = center.x;
	proposed.y = center.y;
	
	return true;
}

void partition(vector<int32_t>& clusters, vector<double>& sortData, size_t left, size_t right){
	
}

void doQuickSort(vector<int32_t>& clusters, vector<double>& sortData, size_t left, size_t right){
	int i = left, j = right;
    int tmp;
    int pivot = sortData[(left + right) / 2];
    
    
}

void clustersQuickSort(vector<int32_t>& clusters, vector<double>& sortData){
	doQuickSort(clusters, sortData, 0, clusters.size());
}

void sortClustersByLength(IntIntListMap* clusterMap, vector<int32_t>& clusters){
	
	vector<double> length;
	bool empty = clusters.empty();
	
	GHashTableIter iter;
	gpointer key;
	gpointer value;
	g_hash_table_iter_init (&iter, clusterMap);

	while (g_hash_table_iter_next (&iter, &key, &value)){
		int32_t* k = (int32_t *)key;
		IntArrayList *lst = (IntArrayList *)value;
		double rd = lst->size;;
		
		if(empty){
			clusters.push_back(*k);
			length.push_back(rd);
		} else{
			
		}
		
		
	}
	
}


void sortClustersByDistance(IntIntListMap* clusterMap, IntDoubleListMap* distancesMap, vector<int32_t>& clusters){
	vector<double> distances;
	
	if(clusters.empty()){
		GHashTableIter iter;
		gpointer key;
		gpointer value;
		g_hash_table_iter_init (&iter, distancesMap);

		while (g_hash_table_iter_next (&iter, &key, &value)){
			int32_t* k = (int32_t *)key;
			DoubleArrayList *lst = (DoubleArrayList *)value;
			double *ddata = (double *)lst->data;
			double rd = ddata[3]/ddata[2];
			
			clusters.push_back(*k);
			distances.push_back(rd);
		}
	}

}

/**
 * 
 * 
 */ 
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
						
			//if(n_rect.x < 0 || n_rect.y < 0 || (n_rect.x + n_rect.width) >= frame.cols || (n_rect.y + n_rect.height) >= frame.rows){
				//cout << "Skipping " << n_rect << endl;
				//continue;
			//}
			//cout << n_rect ;
			//if(!stabiliseRectByMoments(frame, mbs.box, n_rect)){
				//cout << endl;
				//continue;
			//}
			//cout << " stabilised to " << n_rect << endl;
			
			// check that the rect does not already exist
			int idx = rectExist(*boxStructures, n_rect);
			if(idx == -1){
				box_structure bst;
				bst.box = n_rect;
				bst.points.push_back(point);
								
				//cout << mbs.box << " : " << n_rect;
				trimRect(n_rect, frame.rows, frame.cols);
								
				if(n_rect.height < 1 || n_rect.width < 1){
					continue;
				}
				
				double area1 = bst.box.width * bst.box.width; 
				double area2 = n_rect.width * n_rect.width;
				double ratio = area2/area1;
				if(ratio < 0.2){
					//cout << "Ratio is " << ratio<< " Skipping " << n_rect << endl;
					continue;
				}
				
				//cout << " (" << n_rect << ") compare ";
				
				//if(n_rect.x < 0 || ){
				//}
				
				bst.img_ = frame(n_rect);
				calculateHistogram(bst);
				bst.histCompare = compareHist(mbs.hist, bst.hist, CV_COMP_CORREL);
				
				Mat g1, g2;
				cvtColor(mbs.img_, g1, COLOR_RGB2GRAY);
				cvtColor(bst.img_, g2, COLOR_RGB2GRAY);
				bst.momentsCompare = matchShapes(g1, g2, CONTOURS_MATCH_I3, 0);
				//cout << " (" << n_rect << ") compare " << bst.histCompare << " moments compare " << bst.momentsCompare << endl;
				//if(bst.momentsCompare > 0.05){
					//cout << "Skipping for low similarity" << endl;
					//continue;
				//}
								
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
void extendBoxClusters(Mat& frame, results_t* res, set<int32_t>& processedClusters){
	
	GHashTableIter iter;
	gpointer key;
	gpointer value;
	g_hash_table_iter_init (&iter, res->clusterMap);
	vector<box_structure>* boxStructures = res->boxStructures;
	//printf("Originally found %lu objects\n", boxStructures->size());			

	while (g_hash_table_iter_next (&iter, &key, &value)){
		int32_t* kk = (int32_t *)key;
		
		if(processedClusters.find(*kk) != processedClusters.end()){ // Check the clusters that have not already processed
			IntArrayList* list = (IntArrayList *)value;
			vector<int32_t> l1(list->size, -1);
			int first = -1;
			KeyPoint first_kp;
//#pragma omp parallel for	
			for(int32_t i = 0; i < list->size; i++){
				KeyPoint& kp = res->keypoints->at(i);
				for(uint j = 0; j < boxStructures->size(); j++){
					box_structure& stru = boxStructures->at(j);
					if(stru.box.contains(kp.pt)){
						l1[i] = j;
						stru.points.push_back(kp);
						if(first == -1){
							first = j;
						}
						break;
					}
					
					if(first != -1){
						first_kp = kp;
						break;
					}
				}
			}
			box_structure& stru = boxStructures->at(first);
			vector<KeyPoint> kps;
			getListKeypoints(*(res->keypoints), list, kps);
			cout << stru.box << " \n" << stru.hist << endl;
			addToBoxStructure(boxStructures, kps, first_kp, stru, frame);
		}		
	}
	
	//printf("Now has found %lu objects\n", boxStructures->size());
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
		//int32_t label = it->first;
		DoubleArrayList *lst = (DoubleArrayList *)g_hash_table_lookup(res->distancesMap, &(it->first));
		double *ddata = (double *)lst->data;
		double rc = ddata[1]/ddata[0];
		double rd = ddata[3]/ddata[2];
		
		//TODO: Clean this memory leak
		const char* maxCrStr = get_max_cr();
		const char* maxDrStr = get_max_dr();
		double *rcm = (double *)g_hash_table_lookup(res->stats, maxCrStr);
		double *rdm = (double *)g_hash_table_lookup(res->stats, maxDrStr);
		
		double fc = ((*rcm - rc) / (*rcm)) * 100;
		double fd = ((*rdm - rd) / (*rdm)) * 100;
				
		ss += "-";
		ss += to_string((int)fc);
		ss += "-";
		ss += to_string((int)fd);
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
			//printf("Cleaning results %s\n", it->first.c_str());
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

void getBoxStructure(results_t* res, vector<Rect2d>& rois, Mat& frame, bool extend){
	vector<vector<box_structure>> b_structures;
	set<int32_t> processedClusters;
	
	for(map_kp::iterator it = res->finalPointClusters->begin(); it != res->finalPointClusters->end(); ++it){
		processedClusters.insert(it->first);
		IntArrayList *roiPoints = (IntArrayList *)g_hash_table_lookup(res->roiClusterPoints, &(it->first));
		vector<vector<KeyPoint>> kps;
		if(roiPoints->size > 1){
			IntArrayList *cPoints = (IntArrayList *)g_hash_table_lookup(res->clusterMap, &(it->first));
			splitROICluster(roiPoints, cPoints, *(res->dataset), kps, *(res->keypoints));
		} else{
			kps.push_back(it->second);
		}
		
		vector<box_structure> str2;
		box_structure mbs;
		mbs.box = rois[0];
		mbs.img_ = frame(mbs.box);
		calculateHistogram(mbs);
		mbs.histCompare = compareHist(mbs.hist, mbs.hist, CV_COMP_CORREL);
		Mat g1;
		cvtColor(mbs.img_, g1, COLOR_RGB2GRAY);
		mbs.momentsCompare = matchShapes(g1, g1, CONTOURS_MATCH_I3, 0);
		str2.push_back(mbs);
		
		for(vector<vector<KeyPoint> >::iterator itr = kps.begin(); itr != kps.end(); ++itr){
			KeyPoint kp;
			for(vector<KeyPoint>::iterator iter = itr->begin(); iter != itr->end(); ++iter){
				
				if(rois[0].contains(iter->pt)){
					kp = *iter;
					break;
				}
			}
			mbs.points.push_back(kp);
			addToBoxStructure(&str2, it->second, kp, mbs, frame);
		}
		b_structures.push_back(str2);
	}
	
	for(vector<vector<box_structure>>::iterator iter = b_structures.begin(); iter != b_structures.end(); ++iter){
		for(vector<box_structure>::iterator it = iter->begin(); it != iter->end(); ++it){
			int idx = rectExist(*(res->boxStructures), it->box);
			if(idx == -1){
				res->boxStructures->push_back(*it);
			} else{ /// The rect exist s merge the points
				box_structure& strct = res->boxStructures->at(idx);
				strct.points.insert(strct.points.begin(), it->points.begin(), it->points.end());
			}
		}
	}
	
	if(extend){								
		extendBoxClusters(frame, res, processedClusters);
	}
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

void getFrameTruth(String truthFolder, map<int, int>& truth){
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
			truth[fnum] = int(max);
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
		int32_t* k = (int32_t *) key;
		getListKeypoints(*keypoints, clusterLabels, mp[*k]);
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
	hdbscan scan(3, DATATYPE_FLOAT);
	scan.run(dataset.ptr<float>(), dataset.rows, dataset.cols, TRUE);	
	
	for(int i = 3; i < 30; i++){

		printf("\n\n >>>>>>>>>>>> Clustering for minPts = %d\n", i);
			
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
		scan.reRun(i + 1);
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
					//printf("%%%%%%%%%%%%%% added cluster %d of size %d\n", *k, list->size);
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

results_t* do_cluster(results_t* res, Mat& dataset, vector<KeyPoint>& keypoints, int step, int f_minPts, bool analyse, bool singleRun){
	
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
		
		if(singleRun){
			break;
		}
		
		i++;
	}
	res->ogsize = keypoints.size();

	printf("Selected max clustering size = %d and cluster table has %d\n", res->minPts, g_hash_table_size(res->clusterMap));
	
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
    calcHist( &bst.hsv, 1, channels, Mat(), bst.hist, 2, histSize, ranges, true, false );
    normalize( bst.hist, bst.hist, 0, 1, NORM_MINMAX, -1, Mat() );
    //bst.hist = bst.hist.clone();
}
