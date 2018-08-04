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
#include <lmdb.h>
#include <QDir>
#include <math.h>

using namespace std;
int32_t findCenterMostFeature(Rect2d& roi, vector<int32_t>& roiFeatures, vector<KeyPoint>* keypoints);
bool trimRect(Rect& r, int rows, int cols, int padding);
Rect shiftRect(Rect box, Point2f first, Point2f second);
set<int32_t> findValidROIFeature(vector<KeyPoint>& keypoints, Rect2d& roi, vector<int32_t>& roiFeatures, vector<int32_t>* labels);

static COLOURS colours;

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
			circle(x, Point(it->pt.x, it->pt.y), 3, colour, CV_FILLED, 8, 0);
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

/**
 * 
 * 
 */ 
void generateFinalPointClusters(vector<int32_t>& roiFeatures, results_t* res){
	set<int32_t> st;	// cluster for the roi features
	for (vector<int32_t>::iterator it = roiFeatures.begin(); it != roiFeatures.end(); ++it) {
		int k = res->labels->at(*it);
		if(k != 0){		
			vector<int32_t>& list = (*(res->roiClusterPoints))[k];
			list.push_back(*it);
			st.insert(k);
		}
	}
	
	for (set<int32_t>::iterator it = st.begin(); it != st.end(); ++it){
		int32_t key = *it;
		int_array_list_append(res->objectClusters, key);
	}
	
	if(res->objectClusters->size > 0){
		res->objectClusters = hdbscan_sort_by_similarity(res->distancesMap, res->objectClusters, INTRA_DISTANCE_TYPE);
	}
	
	int32_t *data = (int32_t *)res->objectClusters->data;
	for(int32_t i = res->objectClusters->size - 1; i >= 0; i--){		
		IntArrayList* l1 = (IntArrayList*)g_hash_table_lookup(res->clusterMap, data+i);			
		getListKeypoints(*(res->keypoints), l1, (*(res->finalPointClusters))[data[i]]);
	}
}

int rectExist(vector<box_structure>& structures, box_structure& bst){

	double maxIntersect = 0.0;
	int maxIndex = -1;

	for(uint i = 0; i < structures.size(); i++){
		Rect r2 = bst.box & structures[i].box;
		double sect = ((double)r2.area()/bst.box.area()) * 100;
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

/**
 * Trim a rectangle
 */ 
bool trimRect(Rect& r, int rows, int cols, int padding){
	bool trimmed = false;
	if(r.x < padding){
		r.width += r.x - padding;
		r.x = padding;
		trimmed = true;
	}
				
	if(r.y < padding){
		r.height += r.y - padding;
		r.y = padding;
		trimmed = true;
	}
				
	if((r.x + r.width) >= cols - padding){
		r.width = cols - r.x - padding;
		trimmed = true;
	}
				
	if((r.y + r.height) >= rows - padding){
		r.height = rows - r.y - padding;
		trimmed = true;
	}
	
	return trimmed;
}

/**
 * Stabilise the proposed object location by using template matching
 * to find the best possible location of maximum similarity
 */ 
bool stabiliseRect(Mat frame, Rect templ_r, Rect& proposed){
	Mat result;
	
	Rect new_r = proposed;
	int half_h = new_r.height/2;
	int half_w = new_r.width/2;
	new_r.x -= half_w/2;
	new_r.y -= half_h/2;
	new_r.width += half_w; //new_r.width;
	new_r.height += half_h; //new_r.height;
	
	trimRect(new_r, frame.rows, frame.cols, 0);
	if(new_r.height < 1 || new_r.width < 1){
		return false;
	}
	
	Mat img = frame(new_r);
	trimRect(templ_r, frame.rows, frame.cols, 0);
	
	Mat templ = frame(templ_r);
	
	if(img.rows < templ.rows && img.cols < templ.cols){
		return false;
	}
	
	int result_cols =  img.cols - templ.cols + 1;
	int result_rows = img.rows - templ.rows + 1;
		
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
	
	proposed.x = matchLoc.x + new_r.x;
	proposed.y = matchLoc.y + new_r.y;
	
	return true;
}


bool _stabiliseRect(Mat frame, Rect templ_r, Rect& proposed){
	Mat result;
	trimRect(templ_r, frame.rows, frame.cols, 0);
	if(templ_r.height < 1 || templ_r.width < 1){
		return false;
	}
	
	Mat templ = frame(templ_r);
	int result_cols =  frame.cols - templ.cols + 1;
	int result_rows = frame.rows - templ.rows + 1;
	
	if(result_rows < 2 || result_cols < 2){
		
		return false;
	}
	
	Rect rec = proposed;
	result.create( result_rows, result_cols, CV_32FC1 );
	matchTemplate( frame, templ, result, TM_SQDIFF);
	
	trimRect(rec, result.rows, result.cols, 0);
	
	if(rec.height < 1 || rec.width < 1){
		return false;
	}
	Mat p_img = result(rec);
	
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	minMaxLoc( p_img, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
	
	int half_h = p_img.rows/2;
	int half_w = p_img.cols/2;
	Point half_p(half_w, half_h);
	Point diff_p = minLoc - half_p;
	cout << proposed << " : " << minLoc << " - " << half_p  << " = " << diff_p << endl;
	
	proposed.x = proposed.x + diff_p.x;
	proposed.y = proposed.y + diff_p.y;	
	cout << proposed << endl;
	
	return true;
}

bool stabiliseRectByMoments(Mat& frame, const box_structure templ_r, Rect& proposed){
	Mat gray;
	cvtColor(frame, gray, COLOR_RGB2GRAY);
	Rect center = proposed;
	int min = -1;
	double minMom;
	
	do {
		vector<Rect> rects;
		trimRect(center, frame.rows, frame.cols, 0);
		Mat centerImg = gray(center);	
		double momCompare;
		if(min == -1){
			momCompare = matchShapes(templ_r.gray, centerImg, CONTOURS_MATCH_I3, 0);
			minMom = momCompare;
		} else{
			min = -1;
		}
				
		Rect top = center;
		top.y -= 1;
		trimRect(top, frame.rows, frame.cols, 0);
		rects.push_back(top);
				
		Rect bottom = center;
		bottom.y += 1;
		trimRect(bottom, frame.rows, frame.cols, 0);
		rects.push_back(bottom);
				
		Rect right = center;
		right.x += 1;
		trimRect(right, frame.rows, frame.cols, 0);
		rects.push_back(right);
		
		Rect left = center;
		left.x -= 1;
		trimRect(left, frame.rows, frame.cols, 0);
		rects.push_back(left);
		
		Rect topLeft = center;
		topLeft.y -= 1;
		topLeft.x -= 1;
		trimRect(topLeft, frame.rows, frame.cols, 0);
		rects.push_back(topLeft);
				
		Rect bottomLeft = center;
		topLeft.y += 1;
		topLeft.x -= 1;
		trimRect(bottomLeft, frame.rows, frame.cols, 0);
		rects.push_back(bottomLeft);
		
		Rect topRight = center;
		topLeft.y -= 1;
		topLeft.x += 1;
		trimRect(topRight, frame.rows, frame.cols, 0);
		rects.push_back(topRight);
		
		Rect bottomRight = center;
		topLeft.y += 1;
		topLeft.x += 1;
		trimRect(bottomRight, frame.rows, frame.cols, 0);
		rects.push_back(bottomRight);
		
		for(uint i = 0; i < rects.size(); i++){			
			if(rects[i].height < 1 || rects[i].width < 1){
				continue;
			}
			Mat m = gray(rects[i]);
			momCompare = matchShapes(templ_r.gray, m, CONTOURS_MATCH_I3, 0);
			if(momCompare < minMom){
				min = i;
				minMom = momCompare;
			}
		}
		
		if(min != -1){
			center = rects[min];
		}
		
	} while(min != -1);
	
	proposed.x = center.x;
	proposed.y = center.y;
	
	return true;
}

Rect shiftRect(Rect box, Point2f first, Point2f second){
	Rect n_rect = box;;
	Point2f pshift;
	pshift.x = second.x - first.x;
	pshift.y = second.y - first.y;
	//printf("pshift = (%f, %f)\n", pshift.x, pshift.y);
	Point pp = pshift;
	n_rect = n_rect + pp;
	
	return n_rect;
}

/**
 * 
 * 
 */ 
void addToBoxStructure(vector<box_structure>* boxStructures, vector<KeyPoint> c_points, vector<Rect2d>& rects, KeyPoint first_p, box_structure mbs, Mat& frame){
	rects.push_back(mbs.box);
	for(uint j = 0; j < c_points.size(); j++){			
		KeyPoint point = c_points[j];
		if(point.pt != first_p.pt){ // roi points have their own structure "mbs"
			
			//Point2f pshift;
			//pshift.x = point.pt.x - first_p.pt.x;
			//pshift.y = point.pt.y - first_p.pt.y;

			// shift the roi to get roi for a possible new object
			//Rect n_rect = mbs.box;
			
			//Point pp = pshift;
			//n_rect = n_rect + pp;
			
			Rect n_rect = shiftRect(mbs.box, first_p.pt, point.pt);
			stabiliseRect(frame, mbs.box, n_rect);
			
			box_structure bst;
			bst.box = n_rect;
			bst.points.push_back(point);
			
			trimRect(n_rect, frame.rows, frame.cols, 0);
								
			if(n_rect.height < 1 || n_rect.width < 1){
				continue;
			}
				
			double area1 = bst.box.width * bst.box.width; 
			double area2 = n_rect.width * n_rect.width;
			double ratio = area2/area1;
			if(ratio < 0.2){
				continue;
			}
								
			bst.img_ = frame(n_rect);
			calculateHistogram(bst);
			cvtColor(bst.img_, bst.gray, COLOR_RGB2GRAY);
			bst.histCompare = compareHist(mbs.hist, bst.hist, CV_COMP_CORREL);
			
			bst.momentsCompare = matchShapes(mbs.gray, bst.gray, CONTOURS_MATCH_I3, 0);
			
			int idx = rectExist(*boxStructures, bst);
			if(idx == -1){				
				boxStructures->push_back(bst);
				rects.push_back(bst.box);
			} else{
				(*boxStructures)[idx].points.push_back(point);
				rects.push_back((*boxStructures)[idx].box);
			}
		}
	}
}

/**
 * Find clusters that have points inside one of the bounding boxes
 * 
 */ 
set<int32_t> extendBoxClusters(Mat& frame, results_t* res, set<int32_t>& processedClusters){
	GHashTableIter iter;
	gpointer key;
	gpointer value;
	g_hash_table_iter_init (&iter, res->clusterMap);
	vector<box_structure>* boxStructures = res->boxStructures;
	set<int32_t> prcl;		

	while (g_hash_table_iter_next (&iter, &key, &value)){
		int32_t* kk = (int32_t *)key;
		if(*kk != 0 && processedClusters.find(*kk) == processedClusters.end()){ // Check the clusters that have not already processed
			IntArrayList* list = (IntArrayList *)value;
			int32_t* ldata = (int32_t *)list->data;
			int first = -1;
			KeyPoint first_kp;
//#pragma omp parallel for	
			for(int32_t i = 0; i < list->size; i++){
				
				KeyPoint& kp = res->keypoints->at(ldata[i]);
				vector<uint> strsIdx;
				for(uint j = 0; j < boxStructures->size(); j++){
					box_structure& stru = boxStructures->at(j);
					if(stru.box.contains(kp.pt)){
						strsIdx.push_back(j);	
					}
				}
				
				if(!strsIdx.empty()){
					first = strsIdx[0];
					double minMoments = boxStructures->at(strsIdx[0]).momentsCompare;
					for(uint j = 0; j < strsIdx.size(); j++){
						box_structure& stru = boxStructures->at(strsIdx[j]);
						if(minMoments < stru.momentsCompare){
							minMoments = stru.momentsCompare;
							first = strsIdx[j];
						}
					}
					
					boxStructures->at(first).points.push_back(kp);
					first_kp = kp;
					break;
				}
			}
			
			// Some clusters will not interact with the available boxes
			if(first > -1){
				box_structure& stru = boxStructures->at(first);
				vector<KeyPoint>& kps = (*(res->finalPointClusters))[*kk];
				getListKeypoints(*(res->keypoints), list, kps);
				addToBoxStructure(boxStructures, kps, (*res->clusterStructures)[*kk], first_kp, stru, frame);
				prcl.insert(*kk);
			} 
		}		
	}
	
	return prcl;
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

		Mat kimg = drawKeyPoints(frame, it->second, colours.red, -1);
		
		vector<Rect2d>& rects = (*res->clusterStructures)[it->first];
		for(uint i = 0; i < rects.size(); i++){
			
			RNG rng(12345);
			Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
						rng.uniform(0, 255));		
			rectangle(kimg, rects[i], value, 2, 8, 0);
		}
		rectangle(kimg, rects[0], colours.red, 2, 8, 0);
		String ss = "img_keypoints-";
		string s = to_string(it->first);
		ss += s.c_str();
		distance_values *dv = (distance_values *)g_hash_table_lookup(res->distancesMap, &(it->first));
				
		ss += "-";
		ss += to_string((int)dv->cr_confidence);
		ss += "-";
		ss += to_string((int)dv->dr_confidence);
		(*(res->selectedClustersImages))[ss] = kimg;
		kp.insert(kp.end(), it->second.begin(), it->second.end());
	}

	Mat mm = drawKeyPoints(frame, kp, colours.red, -1);

	String ss = "img_allkps";
	(*(res->selectedClustersImages))[ss] = mm;
}

void maintaintHistory(vocount& voc, framed& f){
	voc.frameHistory.push_back(f);
	if(voc.frameHistory.size() > 10){
		framed& f1 = voc.frameHistory.front();
		
		for(map<ResultIndex, results_t*>::iterator it = f1.results.begin(); it != f1.results.end(); ++it){
			cleanResult(it->second);
		}
		
		voc.frameHistory.erase(voc.frameHistory.begin());
	}
}

Mat getDescriptorDataset(vector<framed>& frameHistory, int step, Mat descriptors, vector<KeyPoint> keypoints, bool includeAngle, bool includeOctave){
	Mat dataset = descriptors.clone();
	
	if(includeAngle){
		Mat angles(descriptors.rows, 1, CV_32FC1);
		float* data = angles.ptr<float>(0);

#pragma omp parallel for shared(data)	
		for(size_t i = 0; i < keypoints.size(); i++){
			KeyPoint kp = keypoints[i];
			data[i] = (M_PI / 180) * kp.angle;
		}
		
		hconcat(dataset, angles, dataset);
	}
	
	if(includeOctave){
		Mat octaves(descriptors.rows, 1, CV_32FC1);
		float* data = octaves.ptr<float>(0);
#pragma omp parallel for shared(data)			
		for(size_t i = 0; i < keypoints.size(); i++){
			KeyPoint kp = keypoints[i];
			data[i] = (M_PI/180) * kp.octave;
		}
		
		hconcat(dataset, octaves, dataset);		
	}
	
	if(!dataset.isContinuous()){
		dataset = dataset.clone();
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
void findROIFeature(vector<KeyPoint>& keypoints, Mat& descriptors, Rect2d& roi, vector<int32_t>& roiFeatures, Mat& roiDesc, int32_t& centerFeature){
	Rect2d& r = roi;
	Point2f p;

	p.x = (r.x + r.width)/2.0f;
	p.y = (r.y + r.height)/2.0f;
	double distance;
	for(uint i = 0; i < keypoints.size(); ++i){		
		if(roi.contains(keypoints[i].pt)){
			roiFeatures.push_back(i);
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
			Mat t = descriptors.row(i);
			roiDesc.push_back(t);
			
		}
			
	}
}

int32_t findCenterMostFeature(Rect2d& roi, vector<int32_t>& roiFeatures, vector<KeyPoint>* keypoints){
	int32_t centerFeature = -1;
	Point2f p;

	p.x = (roi.x + roi.width)/2.0f;
	p.y = (roi.y + roi.height)/2.0f;
	double distance;
	
	for(size_t i = 0; i < roiFeatures.size(); i++){
		KeyPoint& r_kp = keypoints->at(roiFeatures[i]);
		if(centerFeature == -1){
			centerFeature = i;
			distance = calcDistanceL1(p, r_kp.pt);
		} else {
			double d1 = calcDistanceL1(p, r_kp.pt);
			if(d1 < distance){
				distance = d1;
				centerFeature = i;
			}
		}
	}
	
	return centerFeature;
}

/**
 * Find the roi features and at the same time find the central feature.
 */
set<int32_t> findValidROIFeature(vector<KeyPoint>& keypoints, Rect2d& roi, vector<int32_t>& roiFeatures, vector<int32_t>* labels){
	Rect2d& r = roi;
	Point2f p;
	set<int32_t> roiClusters;
	p.x = (r.x + r.width)/2.0f;
	p.y = (r.y + r.height)/2.0f;
	for(uint i = 0; i < keypoints.size(); ++i){
		int32_t label = labels->at(i);
		if(roi.contains(keypoints[i].pt) && label != 0){
			roiClusters.insert(label);
			roiFeatures.push_back(i);	
		} 
	}
	
	return roiClusters;
}

/**
 * 
 * 
 */ 
void getBoxStructureByHomography(results_t* res, Rect2d& roi, Mat& frame, bool extend, bool reextend){
	
	//int32_t centerFeature;
	Mat roiDesc;
	//vector<int32_t> roiFeatures;
	vector<int32_t> validObjFeatures;
	
	//cout << "First roi " << roi << endl;
	
	/*findROIFeature(*(res->keypoints), *(res->dataset), roi, roiFeatures, roiDesc, centerFeature);
	
	set<int32_t> roiClusters;
	for(size_t i = 0; i < roiFeatures.size(); i++){
		int32_t label = res->labels->at(roiFeatures[i]);
		if(label != 0){
			roiClusters.insert(label);
			validFeatures.push_back(roiFeatures[i]);
			obj.push_back(res->keypoints->at(roiFeatures[i]).pt);
		}
	}
	roiFeatures.clear();*/
	set<int32_t> roiClusters = findValidROIFeature(*(res->keypoints), roi, validObjFeatures, res->labels);
	//centerFeature = findCenterMostFeature(roi, validObjFeatures, res->keypoints);
		
	cout << "Found " << validObjFeatures.size() << " valid features" << endl;
	
	for(size_t i = 0; i < validObjFeatures.size(); i++){
		int32_t label = res->labels->at(validObjFeatures[i]);
		KeyPoint r_kp = res->keypoints->at(validObjFeatures[i]);
		
		IntArrayList* p = (IntArrayList *)g_hash_table_lookup (res->clusterMap, &label);
		int32_t* data = (int32_t *)p->data;
		///
		/// Create a scene for each of the points in the cluster
		for(int32_t j = 0; j < p->size; j++){
			KeyPoint n_kp = res->keypoints->at(data[j]);
			Rect newRect = shiftRect(roi, r_kp.pt, n_kp.pt);
			//cout << "roi is " << roi << endl;
			//printf("first -> (%f, %f), second -> (%f, %f)\n", r_kp.pt.x, r_kp.pt.y, n_kp.pt.x, n_kp.pt.y);
			//cout << "newRect = " << newRect << endl;
			// widen the new rect
			int32_t h_p = newRect.height/2;
			int32_t w_p = newRect.width/2;
			
			newRect.x = newRect.x - w_p;
			newRect.y = newRect.y - h_p;
			newRect.height = newRect.height * 2;
			newRect.width = newRect.width * 2;
			
			//cout << "Finding features at " << newRect << endl;
			bool trimmed = trimRect(newRect, frame.rows, frame.cols, 0);
			Rect2d rr = Rect2d(newRect);
			//cout << "Finding features at " << rr << endl;
			vector<int32_t> sceneFeatures;
			findValidROIFeature(*(res->keypoints), rr, sceneFeatures, res->labels);
			vector<int32_t> validSceneFeatures;
			
			// Find scene points that share clusters with object points
			for(size_t k = 0; k < sceneFeatures.size(); k++){
				int32_t lbl = res->labels->at(sceneFeatures[k]);
				set<int32_t>::iterator itr = roiClusters.find(lbl);
				if(itr != roiClusters.end()){
					validSceneFeatures.push_back(sceneFeatures[k]);
				}
			}
			
			cout << validSceneFeatures.size() << " - " << validObjFeatures.size() << endl;
			
			// create matching points for scene and object
			// For each of the points in the object, use cluster label
			// to match with any of the points in the scene
			vector<Point2f> scene;
			vector<Point2f> obj;
			for(size_t k = 0; k < validObjFeatures.size(); k++){
				int32_t lbl = res->labels->at(validObjFeatures[k]);
				for(size_t l = 0; l < validSceneFeatures.size(); l++){
					int32_t lbl2 = res->labels->at(validSceneFeatures[l]);
					
					if(lbl == lbl2){
						obj.push_back(res->keypoints->at(validObjFeatures[k]).pt);
						scene.push_back(res->keypoints->at(validSceneFeatures[l]).pt);
						printf("(%f, %f) to (%f, %f)\n", res->keypoints->at(validObjFeatures[k]).pt.x, 
														res->keypoints->at(validObjFeatures[k]).pt.y, 
														res->keypoints->at(validSceneFeatures[l]).pt.x, 
														res->keypoints->at(validSceneFeatures[l]).pt.y);
					}
				}
			}
						
			printf("obj.size() = %ld, scene.size() = %ld\n", obj.size(), scene.size());
			if(obj.size() > 3){
				Mat H = findHomography( obj, scene, 0 );
				Mat img_object = frame(newRect);
				cout<< endl << H << endl << endl;
				//-- Get the corners from the image_1 ( the object to be "detected" )
				std::vector<Point2f> obj_corners(4);
				obj_corners[0] = cvPoint(0,0); 
				obj_corners[1] = cvPoint( img_object.cols, RHO);
				obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); 
				obj_corners[3] = cvPoint( 0, img_object.rows );
				
				vector<Point2f> scene_corners(4);
				
				if(!H.empty()){
					perspectiveTransform( obj_corners, scene_corners, H);
					
					cout << scene_corners[0].x << ", " << scene_corners[0].y << endl;
					cout << scene_corners[1].x << ", " << scene_corners[1].y  << endl;
					cout << scene_corners[2].x << ", " << scene_corners[2].y  << endl;
					cout << scene_corners[3].x << ", " << scene_corners[3].y  << endl << endl << endl;	
					
					//-- Draw lines between the corners (the mapped object in the scene - image_2 )
					line( frame, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
					line( frame, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
					line( frame, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
					line( frame, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
					printImage("out", 222, "mymatches", frame);
				}
			}	
		}
	}
	exit(0);
}

/**
 * 
 * 
 */ 
void getBoxStructure(results_t* res, Rect2d& roi, Mat& frame, bool extend, bool reextend){
	vector<vector<box_structure>> b_structures;
	set<int32_t> processedClusters;
	int32_t *data = (int32_t *)res->objectClusters->data;
	
	for(int32_t i = res->objectClusters->size - 1; i >= 0; i--){
		int32_t key = data[i];
		processedClusters.insert(key);
		(*res->clusterStructures)[key];
		vector<vector<KeyPoint>> kps;
		kps.push_back(res->finalPointClusters->at(key));
		
		vector<box_structure> str2;
		box_structure mbs;
		mbs.box = roi;
		mbs.img_ = frame(mbs.box);
		calculateHistogram(mbs);
		mbs.histCompare = compareHist(mbs.hist, mbs.hist, CV_COMP_CORREL);
		cvtColor(mbs.img_, mbs.gray, COLOR_RGB2GRAY);
		mbs.momentsCompare = matchShapes(mbs.gray, mbs.gray, CONTOURS_MATCH_I3, 0);
		str2.push_back(mbs);
		
		for(vector<vector<KeyPoint> >::iterator itr = kps.begin(); itr != kps.end(); ++itr){
			KeyPoint kp;
			for(vector<KeyPoint>::iterator iter = itr->begin(); iter != itr->end(); ++iter){
				
				if(roi.contains(iter->pt)){
					kp = *iter;
					break;
				}
			}
			mbs.points.push_back(kp);
			addToBoxStructure(&str2, res->finalPointClusters->at(key), res->clusterStructures->at(key), kp, mbs, frame);
		}
		b_structures.push_back(str2);
		
	}
	
	for(vector<vector<box_structure>>::iterator iter = b_structures.begin(); iter != b_structures.end(); ++iter){
		for(vector<box_structure>::iterator it = iter->begin(); it != iter->end(); ++it){
			int idx = rectExist(*(res->boxStructures), *it);
			if(idx == -1){
				res->boxStructures->push_back(*it);
			} else{ /// The rect exist s merge the points
				box_structure& strct = res->boxStructures->at(idx);
				if(it->histCompare > strct.histCompare){
					strct.box = it->box;
				}
				//printf("comparing (%f, %f) and (%f, %f)\n", it->histCompare, strct.histCompare, it->momentsCompare, strct.momentsCompare);
				strct.points.insert(strct.points.begin(), it->points.begin(), it->points.end());
			}
		}
	}
	
	if(extend){								
		set<int32_t> extras = extendBoxClusters(frame, res, processedClusters);
		
		//if(reextend){
			//extras.insert(processedClusters.begin(), processedClusters.end());
			//extendBoxClusters(frame, res, extras);
		//}
	}
	printf("boxStructure found %lu objects\n\n", res->boxStructures->size());
}

/**
 * Given a vector of box structures, the function draws the rectangles around the identified object locations
 * 
 */ 
void createBoxStructureImages(vector<box_structure>* boxStructures, map<String, Mat>* selectedClustersImages){
	
	String ss = "img_bounds";

	Mat img_bounds = (*selectedClustersImages)["img_allkps"].clone();
//#pragma omp parallel for shared(img_bounds)
	for (size_t i = 0; i < boxStructures->size(); i++) {
		box_structure b = (*boxStructures)[i];
		RNG rng(12345);
		Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
				rng.uniform(0, 255));
//#pragma omp barrier
		rectangle(img_bounds, b.box, value, 2, 8, 0);
	}
	(*selectedClustersImages)[ss] = img_bounds;
}

void getFrameTruth(String truthFolder, map<int, int>& truth){
			
	int rc;
	MDB_env *env;
	MDB_dbi dbi;
	MDB_val key, data;
	MDB_txn *txn;
	MDB_cursor *cursor;
	
	rc = mdb_env_create(&env);
	rc = mdb_env_open(env, truthFolder.c_str(), 0, 0664);
	rc = mdb_txn_begin(env, NULL, MDB_RDONLY, &txn);
	rc = mdb_dbi_open(txn, NULL, 0, &dbi);
	rc = mdb_cursor_open(txn, dbi, &cursor);
	
	while ((rc = mdb_cursor_get(cursor, &key, &data, MDB_NEXT)) == 0) {
		
		String k((char *)key.mv_data);
		k = k.substr(0, key.mv_size);
		
		String v((char *)data.mv_data);
		v = v.substr(0, data.mv_size);
		truth[stoi(k)] = stoi(v);
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
	Mat tmpf;
	GaussianBlur(f, tmpf, Size( 5, 5), 0, 0 );
	float* data = m.ptr<float>(0);
	for(size_t i = 0; i < pts.size(); i++){
		Point2f pt = pts[i].pt;
		
		Vec3b p = tmpf.at<Vec3b>(pt);
		int idx = i * 3;
		
		data[idx] = p.val[0];
		data[idx + 1] = p.val[1];
		data[idx + 2] = p.val[2];
	}
	return m;
}

void getSelectedKeypointsDescriptors(Mat& desc, IntArrayList* indices, Mat& out){
	int32_t *dt = (int32_t *)indices->data;
	for(int i = 0; i < indices->size; i++){
		out.push_back(desc.row(dt[i]));
	}
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
 * Function to assess whether the list of minPts contains a continuous 
 * series of numbers ie: each value m is preceeded by a value of m-1 and
 * preceeds a value of m+1
 * 
 * @param minPts - The set of minPts that result in the same number of
 * 				   clusters
 * @return where the set is continuous or not.
 */ 
bool isContinuous(set<int>& minPtsList){
	bool continous = true;
	int prev = 2;
	for(int m : minPtsList){
		if(prev != 2){
			if(m - prev > 1){
				continous = false;
				break;
			}
		}
		
		prev = m;
	}
	
	return continous;
}

/**
 * Determines whether there are more valid clustering results than 
 * invalid clusters.
 * 
 * @param minPtsList: the list of clusters
 * @param validities: the vector containing validities of all clustering
 * 					  results 
 */ 
bool isValid(set<int>& minPtsList, vector<int>& validities){
	int validCount = 0;
	int invalidCount = 0;
	
	for(int m : minPtsList){
		int validity = validities[m - 3]; // minPts begins at 3
		
		if(validity >= 0){
			validCount++;
		} else if(validity == -2){
			
			return false;		// If any validity is -2 then the sequence is invalid		
		} else{
			invalidCount++;
		}
	}
	
	return validCount > invalidCount;
}

int findLargestSet(map<uint, set<int>>& numClusterMap){
	
	uint numClusters = 0;
	
	/**
	 * Find the first largest sequence of clustering results with the
	 * same number of clusters.
	 */ 
	for(map<uint, set<int>>::iterator it = numClusterMap.begin(); it != numClusterMap.end(); ++it){
		
		if(numClusters == 0 || it->second.size() > numClusterMap.at(numClusters).size()){
			numClusters = it->first;
		} 
		/// If the current largest is equal to the new size, then compare 
		/// the first entries
		else if(it->second.size() == numClusters){
			
			set<int>& previous = numClusterMap.at(numClusters);
			set<int>& current = it->second;
			
			int pfirst = *(previous.begin());
			int cfirst = *(current.begin());
			
			if(cfirst > pfirst){
				numClusters = it->first;
			}
		}
	}
	
	return numClusters;
}

/**
 * 
 * 
 */ 
int chooseMinPts(map<uint, set<int>>& numClusterMap, vector<int>& validities){
	
	uint numClusters = findLargestSet(numClusterMap);
		
	bool found = false;
	while(!found){
		cout << "DEBUG: Largest set is " << numClusters << endl;
		set<int>& currentSelection = numClusterMap.at(numClusters);
		
		found = isContinuous(currentSelection);		
		/*if(!found){				
			cout << "DEBUG: Set for " << numClusters << " is not continuous" << endl;
		}*/
		
		found = found && isValid(currentSelection, validities);
		/*if(!found){
			cout << "DEBUG: Set for " << numClusters << " is not valid" << endl;
		}*/
			
		if(!found){
			map<uint, set<int>> tmp;
			int cfirst = *(currentSelection.begin());
			
			for(map<uint, set<int>>::iterator it = numClusterMap.begin(); it != numClusterMap.end(); ++it){
				set<int>& tmp_set = it->second;
				int tfirst = *(tmp_set.begin());
				if(it->first > numClusters && tfirst < cfirst){
					tmp[it->first] = tmp_set;
				}
			}
			
			numClusters = findLargestSet(tmp);
		}
	}
	
	set<int> sel = numClusterMap.at(numClusters);
	
	return *(sel.begin());
}

/**
 *
 */
selection_t trainColourModel(Mat& frame, Mat& descriptors, vector<KeyPoint>& keypoints, ofstream& trainingFile, ofstream& trackingFile){
	
	map<uint, set<int>> numClusterMap;
	vector<int> validities;
	vector<int32_t*> labelsList;
	map<int, IntIntListMap* > clusterMaps;
	
	printf("Detecting minPts value for colour clustering.\n");
	Mat dataset = getColourDataset(frame, keypoints);
	//printMatToFile(dataset, "out", "colour_training_dset", Formatter::FMT_CSV);
	//printMatToFile(dataset, "out", "colour_training_dset", Formatter::FMT_MATLAB);
	//printMatToFile(dataset, "out", "colour_training_dset", Formatter::FMT_NUMPY);
	
	selection_t colourSelection;
	hdbscan scan(3, DATATYPE_FLOAT);
	scan.run(dataset.ptr<float>(), dataset.rows, dataset.cols, TRUE);	
	
	for(int i = 3; i <= 30; i++){
		if(i > 3){
			scan.reRun(i);
		}
		printf("\n\n >>>>>>>>>>>> Clustering for minPts = %d\n", i);			
		IntIntListMap* clusterMap = hdbscan_create_cluster_table(scan.clusterLabels, 0, scan.numPoints);
		clusterMaps[i] = clusterMap;
		
		//hdbscan_print_cluster_sizes(clusterMap);
		cout << endl << endl;
		IntDistancesMap* distancesMap = hdbscan_get_min_max_distances(&scan, clusterMap);
		clustering_stats stats;
		hdbscan_calculate_stats(distancesMap, &stats);
		int val = hdbscan_analyse_stats(&stats);
		/*cout << "---------------------------------- Stats Values ---------------------------------" << endl;
		cout << "coreDistanceValues >> {";
		cout << "\tMean : " << stats.coreDistanceValues.mean << endl;
		cout << "\tStandard Dev : " << stats.coreDistanceValues.standardDev << endl;
		cout << "\tVariance : " << stats.coreDistanceValues.variance << endl;
		cout << "\tMax : " << stats.coreDistanceValues.max << endl;
		cout << "\tKurtosis : " << stats.coreDistanceValues.kurtosis << endl;
		cout << "\tSkewness : " << stats.coreDistanceValues.skewness << endl;
		cout << "}" << endl << endl; 
		
		cout << "intraDistanceValues >> {";
		cout << "\tMean : " << stats.intraDistanceValues.mean << endl;
		cout << "\tStandard Dev : " << stats.intraDistanceValues.standardDev << endl;
		cout << "\tVariance : " << stats.intraDistanceValues.variance << endl;
		cout << "\tMax : " << stats.intraDistanceValues.max << endl;
		cout << "\tKurtosis : " << stats.intraDistanceValues.kurtosis << endl;
		cout << "\tSkewness : " << stats.intraDistanceValues.skewness << endl;
		cout << "}" << endl << endl; 
		cout << "---------------------------------------------------------------------------------" << endl;
		*/
		
		/*GHashTableIter iter;
		gpointer key;
		gpointer value;
		g_hash_table_iter_init (&iter, distancesMap);

		while (g_hash_table_iter_next (&iter, &key, &value)){
			int32_t* k = (int32_t *)key;
			distance_values* dvs = (distance_values *)value;
			cout << *k << ":";
			cout << "\tmin_cr:" << dvs->min_cr;
			cout << "\tmax_cr:" << dvs->max_cr;
			cout << "\tcr_confidence:" << dvs->cr_confidence;
			cout << "\tmin_dr:" << dvs->min_dr;
			cout << "\tmax_dr:" << dvs->max_dr;
			cout << "\tdr_confidence:" << dvs->dr_confidence << endl;
		}*/
				
		uint idx = g_hash_table_size(clusterMap) - 1;
		int k = 0;
		IntArrayList* p = (IntArrayList *)g_hash_table_lookup (clusterMap, &k);
		
		if(p == NULL){
			idx++;
		}
		
		printf("cluster map has size = %d and validity = %d\n", g_hash_table_size(clusterMap), val);
		if(trainingFile.is_open()){
			int32_t ps = 0;
			
			if(p != NULL){
				ps = p->size;
			}
			
			trainingFile << i << "," << idx << "," << ps << "," << val << "\n";
		}
		
		numClusterMap[idx].insert(i);				
		validities.push_back(val);
		
		int32_t *labelsCopy = new int32_t[scan.numPoints];
		
		for(uint j = 0; j < scan.numPoints; j++){
			labelsCopy[j] = scan.clusterLabels[j];
		}
		
		labelsList.push_back(labelsCopy);
		hdbscan_destroy_distance_map_table(distancesMap);		
	}
		
	colourSelection.minPts = chooseMinPts(numClusterMap, validities);
	
	//printLabelsToFile(labelsList[colourSelection.minPts - 3], scan.numPoints, "out", Formatter::FMT_CSV);
	//printLabelsToFile(labelsList[colourSelection.minPts - 3], scan.numPoints, "out", Formatter::FMT_MATLAB);
	//printLabelsToFile(labelsList[colourSelection.minPts - 3], scan.numPoints, "out", Formatter::FMT_NUMPY);
	
	for(map<int, IntDoubleListMap* >::iterator it = clusterMaps.begin(); it != clusterMaps.end(); ++it){
		if(it->first == colourSelection.minPts){
			colourSelection.clusterKeypointIdx = it->second;
			colourSelection.validity = validities[it->first - 3];
			colourSelection.numClusters = g_hash_table_size(colourSelection.clusterKeypointIdx);
			
		} else{
			hdbscan_destroy_distance_map_table(it->second);			
		}
		delete[] labelsList[it->first - 3];
	}
		
	printf(">>>>>>>> VALID CHOICE OF minPts IS %d <<<<<<<<<\n", colourSelection.minPts);
	if(trainingFile.is_open()){
		trainingFile << "Selected minPts," << colourSelection.minPts << "\n";
		trainingFile.close();
	}
	
	GHashTableIter iter;
	gpointer key;
	gpointer value;
	g_hash_table_iter_init (&iter, colourSelection.clusterKeypointIdx);

	while (g_hash_table_iter_next (&iter, &key, &value)){
		IntArrayList* list = (IntArrayList*)value;
		int32_t* k = (int32_t *)key;
		vector<KeyPoint> kps;
		getListKeypoints(keypoints, list, kps);
		Mat m = drawKeyPoints(frame, kps, colours.red, -1);
		// print the choice images
		String imName = "choice_cluster_";
		imName += std::to_string(*k).c_str();	
		
		if(*k != 0){
			String windowName = "Choose ";
			windowName += std::to_string(*k).c_str();
			windowName += "?";
			display(windowName.c_str(), m);
			
			
			// Listen for a key pressed
			char c = ' ';
			while(true){
				if (c == 'a') {
					cout << "Chosen cluster " << *k << endl;
					Mat xx ;
					getSelectedKeypointsDescriptors(descriptors, list, xx);
					colourSelection.selectedClusters.insert(*k);
					colourSelection.selectedKeypoints.insert(colourSelection.selectedKeypoints.end(), kps.begin(), kps.end());
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
			destroyWindow(windowName.c_str());
		}
	}
	
	if(trackingFile.is_open()){
		trackingFile << 1 << "," << keypoints.size() << "," << colourSelection.selectedDesc.rows << "," << colourSelection.minPts << "," << colourSelection.numClusters << "," << colourSelection.validity << endl;
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
	res->clusterStructures = new map<int32_t, vector<Rect2d>>();
	res->odata = new map<OutDataIndex, int32_t>();
	res->labels = new vector<int32_t>(res->keypoints->size());
	res->boxStructures = new vector<box_structure>();
	res->cest = new vector<int32_t>();
	res->selectedClustersImages = new map<String, Mat>();
	res->leftoverClusterImages = new map<String, Mat>();
	res->objectClusters = int_array_list_init();
		
    res->clusterMap = NULL;		 								/// maps labels to the keypoint indices
    
    res->roiClusterPoints = new map<int32_t, vector<int32_t>>(); /// cluster labels for the region of interest mapped to the roi points in the cluster
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
	
	if(res == NULL){
		res = initResult_t(dataset, keypoints);
	}
	
	int m_pts = step * f_minPts;
	hdbscan scan(m_pts, DATATYPE_FLOAT);
	scan.run(res->dataset->ptr<float>(), res->dataset->rows, res->dataset->cols, TRUE);
	
	IntIntListMap* c_map = NULL;
	IntDistancesMap* d_map = NULL;
	clustering_stats stats;
	int val = -1;
	
	int i = 0;
	
	while(val <= 2 && i < 5){
		
		if(m_pts > (step * f_minPts)){	
			scan.reRun(m_pts);
		}
		
		c_map = hdbscan_create_cluster_table(scan.clusterLabels, 0, keypoints.size());
		
		if(analyse){
			d_map = hdbscan_get_min_max_distances(&scan, c_map);
			hdbscan_calculate_stats(d_map, &stats);
			val = hdbscan_analyse_stats(&stats);
		}
		
		if(c_map != NULL){
			uint hsize = res->clusterMap == NULL ? 0 : g_hash_table_size(res->clusterMap);
			if(g_hash_table_size(c_map) > hsize || val > res->validity){
				if(res->clusterMap != NULL){
					hdbscan_destroy_cluster_table(res->clusterMap);
				}
				
				if(res->distancesMap != NULL){
					hdbscan_destroy_distance_map_table(res->distancesMap);
				}
				
				if(!(res->labels->empty())){
					res->labels->clear();
				}	
				
				res->clusterMap = c_map;
				res->distancesMap = d_map;
				res->stats = stats;
				res->validity = val;
				res->minPts = m_pts;
				res->labels->insert(res->labels->begin(), scan.clusterLabels, scan.clusterLabels + keypoints.size());
		
			} else {
				hdbscan_destroy_cluster_table(c_map);
				hdbscan_destroy_distance_map_table(d_map);
			}
		}
		
		if(singleRun){
			break;
		}
		printf("Testing minPts = %d with validity = %d and cluster map size = %d\n", m_pts, val, g_hash_table_size(c_map));
		i++;
		m_pts = (f_minPts + i) * step;
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
		
		if(res->distancesMap != NULL){
			hdbscan_destroy_distance_map_table(res->distancesMap);
			res->distancesMap = NULL;
		}
				
		delete res->roiClusterPoints;
		delete res->dataset;
		delete res->keypoints;
		delete res->finalPointClusters;
		delete res->odata;
		delete res->labels;
		delete res->boxStructures;
		delete res->cest;
		delete res->selectedClustersImages;
		delete res->leftoverClusterImages;
		delete res->clusterStructures;
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

cv::Ptr<cv::Tracker> createTrackerByName(cv::String name)
{
    cv::Ptr<cv::Tracker> tracker;

    if (name == "KCF")
        tracker = cv::TrackerKCF::create();
    else if (name == "TLD")
        tracker = cv::TrackerTLD::create();
    else if (name == "BOOSTING")
        tracker = cv::TrackerBoosting::create();
    else if (name == "MEDIAN_FLOW")
        tracker = cv::TrackerMedianFlow::create();
    else if (name == "MIL")
        tracker = cv::TrackerMIL::create();
    else if (name == "GOTURN")
        tracker = cv::TrackerGOTURN::create();
    else
        CV_Error(cv::Error::StsBadArg, "Invalid tracking algorithm name\n");

    return tracker;
}

void findNewROIs(Mat& frame, vector<Ptr<Tracker>>& trackers, vector<Rect2d>& newRects, vector<box_structure>* boxStructures, String trackerName){

#pragma omp parallel for
	for(size_t i = 0; i < boxStructures->size(); i++){
		
		double maxIntersect = 0.0;
		//int maxIndex = -1;
		box_structure& bs = boxStructures->at(i);
		
		for(size_t j = 0; j < newRects.size(); j++){
			Rect r1 = newRects[j];
			Rect r2 = r1 & bs.box;
			double sect = ((double)r2.area()/r1.area()) * 100;
			if(sect > maxIntersect){
				//maxIndex = i;
				maxIntersect = sect;
			}
		}
		
		bool valid = 0 <= bs.box.x && 0 <= bs.box.width 
						&& bs.box.x + bs.box.width <= frame.cols 
						&& 0 <= bs.box.y && 0 <= bs.box.height 
						&& bs.box.y + bs.box.height <= frame.rows;
		#pragma omp critical
		if(maxIntersect < 95.0 && valid){
			size_t x = newRects.size();
			newRects.push_back(bs.box);
			trackers.push_back(createTrackerByName(trackerName));
			trackers[x]->init( frame, newRects[x] );
		}

	}
}

Mat do_templateMatch(Mat& frame, Rect2d roi){
	Mat result;
	
	Mat templ = frame(roi);
	
	int result_cols =  frame.cols - templ.cols + 1;
	int result_rows = frame.rows - templ.rows + 1;
	
	result.create( result_rows, result_cols, CV_32FC1 );
	matchTemplate(frame, templ, result, TM_SQDIFF);
	normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
	
	return result;
}

void trackFrameColourModel(vocount& vcount, framed& f, selection_t& colourSel, Mat& frame){
	
	vector<KeyPoint> keyp;
	size_t p_size = 0;
	
	framed ff;	
	Mat dataset; 
	if(!vcount.frameHistory.empty()){
		ff = vcount.frameHistory[vcount.frameHistory.size()-1];
		keyp.insert(keyp.end(), ff.keypoints.begin(), ff.keypoints.end());
		dataset = getColourDataset(ff.frame, keyp);
		p_size = ff.keypoints.size();
	}
	
	keyp.insert(keyp.end(), f.keypoints.begin(), f.keypoints.end());
	dataset.push_back(getColourDataset(frame, f.keypoints));
	dataset = dataset.clone();
	hdbscan scanis(2*colourSel.minPts, DATATYPE_FLOAT);
	scanis.run(dataset.ptr<float>(), dataset.rows, dataset.cols, true);
	
	/****************************************************************************************************/
	/// Get the hash table for the current dataset and find the mapping to clusters in prev frame
	/// and map them to selected colour map
	/****************************************************************************************************/
	IntIntListMap* prevHashTable = colourSel.clusterKeypointIdx;
	int32_t prevNumClusters = colourSel.numClusters;
	colourSel.clusterKeypointIdx = hdbscan_create_cluster_table(scanis.clusterLabels + p_size, 0, f.keypoints.size());
	colourSel.numClusters = g_hash_table_size(colourSel.clusterKeypointIdx);
	IntDoubleListMap* distancesMap = hdbscan_get_min_max_distances(&scanis, colourSel.clusterKeypointIdx);
	clustering_stats stats;
	hdbscan_calculate_stats(distancesMap, &stats);
	int val = hdbscan_analyse_stats(&stats);
	
	if(val < 0){
		cout << "Validity is less than 0. Re clustering ..." << endl;
		hdbscan_destroy_distance_map_table(distancesMap);
		hdbscan_destroy_cluster_table(colourSel.clusterKeypointIdx);
		
		scanis.reRun(2*colourSel.minPts - 1);
		prevNumClusters = colourSel.numClusters;
		colourSel.clusterKeypointIdx = hdbscan_create_cluster_table(scanis.clusterLabels + p_size, 0, f.keypoints.size());
		colourSel.numClusters = g_hash_table_size(colourSel.clusterKeypointIdx);
		distancesMap = hdbscan_get_min_max_distances(&scanis, colourSel.clusterKeypointIdx);
		clustering_stats stats;
		hdbscan_calculate_stats(distancesMap, &stats);
		val = hdbscan_analyse_stats(&stats);
	}
	
	printf("------- MinPts = %d - new validity = %d (%d) and old validity = %d (%d)\n", colourSel.minPts, val, colourSel.numClusters, colourSel.validity, prevNumClusters);
	
	colourSel.validity = val;				
	set<int32_t> currSelClusters, cClusters;
					
	for (set<int32_t>::iterator itt = colourSel.selectedClusters.begin(); itt != colourSel.selectedClusters.end(); ++itt) {
		int32_t cluster = *itt;
		IntArrayList* list = (IntArrayList*)g_hash_table_lookup(prevHashTable, &cluster);
		int32_t* ldata = (int32_t*)list->data;
						
		/**
		 * Since I have no idea whether the clusters from the previous frames will be clustered in the same manner
		 * I have to get the cluster with the largest number of points from selected clusters
		 **/ 
		map<int32_t, vector<int32_t>> temp;
		for(int32_t x = 0; x < list->size; x++){
			int32_t idx = ldata[x];
			int32_t newCluster = (scanis.clusterLabels)[idx];
			temp[newCluster].push_back(idx);
		}
						
		int32_t selC = -1;
		size_t mSize = 0;
		for(map<int32_t, vector<int32_t>>::iterator it = temp.begin(); it != temp.end(); ++it){
			if(mSize < it->second.size()){
				selC = it->first;
				mSize = it->second.size();
			}
			
		}
		currSelClusters.insert(selC);			
	}
			
	// Need to clear the previous table map
	hdbscan_destroy_cluster_table(prevHashTable);
	colourSel.selectedClusters = currSelClusters;
	colourSel.selectedKeypoints.clear();
	colourSel.roiFeatures.clear();
	colourSel.oldIndices.clear();				
				
	/****************************************************************************************************/
	/// Image space clustering
	/// -------------------------
	/// Create a dataset from the keypoints by extracting the colours and using them as the dataset
	/// hence clustering in image space
	/****************************************************************************************************/
					
	Mat selDesc;
	for (set<int32_t>::iterator itt = colourSel.selectedClusters.begin(); itt != colourSel.selectedClusters.end(); ++itt) {
		
		int cluster = *itt;
		IntArrayList* list = (IntArrayList*)g_hash_table_lookup(colourSel.clusterKeypointIdx, &cluster);
		int32_t* ldata = (int32_t*)list->data;
		colourSel.oldIndices.insert(colourSel.oldIndices.end(), ldata, ldata + list->size);
		getListKeypoints(f.keypoints, list, colourSel.selectedKeypoints);
		getSelectedKeypointsDescriptors(f.descriptors, list, selDesc);
			
	}	
	cout << "Selected " << selDesc.rows << " points" << endl;
	
	if(vcount.trackingFile.is_open()){
		vcount.trackingFile << f.i << "," << f.keypoints.size() << "," << selDesc.rows << "," << colourSel.minPts << "," << colourSel.numClusters << "," << val << endl;
	}		
	colourSel.selectedDesc = selDesc.clone();
}

void getROI(vocount& vcount, vsettings& settings, framed& f, Mat& frame){
	if (settings.selectROI && !vcount.roiExtracted) { // select a roi if c has been pressed or if the program was run with -s option
				
		Mat f2 = frame.clone();
		f.roi = selectROI("Select ROI", f2);
		destroyWindow("Select ROI");
				
		vcount.tracker = createTrackerByName(settings.trackerAlgorithm);
		vcount.tracker->init( frame, f.roi);
			
		vcount.roiExtracted = true;
		settings.selectROI = false;
	} else if(vcount.roiExtracted && vcount.roi.area() > 0){
			
		f.roi = vcount.roi;
		vcount.tracker = createTrackerByName(settings.trackerAlgorithm);
		vcount.tracker->init( frame, vcount.roi );
		vcount.roi = Rect2d(0,0,0,0);
	}

	if (vcount.roiExtracted ){
		
		vcount.tracker->update(frame, f.roi);
		findROIFeature(f.keypoints, f.descriptors, f.roi, f.roiFeatures, f.roiDesc, f.centerFeature);
		Rect r = f.roi;
		double d1 = r.area();
		trimRect(r, frame.rows, frame.cols, 10);
		double d2 = r.area();
					
		int rdx = 1;
		int xz = 1;
		/**
		 * select a new roi as long as either d2 < d1 or 
		 * no roi features were found
		 */ 
		while(d2 < d1 || f.roiFeatures.empty()){
			f.roiFeatures.clear();
			f.roiDesc = Mat();
			f.centerFeature = -1;
			
			vector<box_structure>* bxs = vcount.frameHistory[vcount.frameHistory.size()-xz].results.at(ResultIndex::Descriptors)->boxStructures;
			
			if(!bxs->empty()){	
				f.roi = bxs->at(rdx).box;
				Rect prev = f.roi;
				Rect nRect = f.roi;
				stabiliseRect(frame, prev, nRect);
				
				vcount.tracker = createTrackerByName(settings.trackerAlgorithm);
				vcount.tracker->init(vcount.frameHistory[vcount.frameHistory.size()-1].frame, f.roi);
				vcount.tracker->update(frame, f.roi);				
				r = f.roi;
				d1 = r.area();
				trimRect(r, frame.rows, frame.cols, 10);
				
				d2 = r.area();
				findROIFeature(f.keypoints, f.descriptors, f.roi, f.roiFeatures, f.roiDesc, f.centerFeature);
			}else{
				xz++;
			}
			rdx++;
		}
		
		RNG rng(12345);
		Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),	rng.uniform(0, 255));		
		rectangle(frame, f.roi, value, 2, 8, 0);
	}
}

void imageSpaceClustering(vocount& vcount, vsettings& settings, selection_t& colourSel, framed& f){
	
	results_t* res1 = f.results[ResultIndex::Descriptors];
	Mat ds = getImageSpaceDataset(colourSel.selectedKeypoints);
	results_t* idxClusterRes = do_cluster(NULL, ds, colourSel.selectedKeypoints, 1, 3, true, true);
	set<int> ss(idxClusterRes->labels->begin(), idxClusterRes->labels->end());
	printf("We found %lu objects by index points clustering.\n", ss.size() - 1);
	getKeypointMap(idxClusterRes->clusterMap, &colourSel.selectedKeypoints, *(idxClusterRes->finalPointClusters));
	generateClusterImages(f.frame, idxClusterRes);
	f.results[ResultIndex::ImageSpace] = idxClusterRes;		
		
	if(settings.print){
		String iSpaceFrameDir = createDirectory(settings.imageSpaceDir, to_string(vcount.frameCount));					
		generateOutputData(vcount, f.frame, colourSel.selectedKeypoints, colourSel.roiFeatures, idxClusterRes, f.i);
		Mat frm = drawKeyPoints(f.frame, colourSel.selectedKeypoints, colours.red, -1);
		printImage(settings.imageSpaceDir, vcount.frameCount, "frame_kp", frm);
		printImages(iSpaceFrameDir, idxClusterRes->selectedClustersImages, vcount.frameCount);
		printEstimates(vcount.indexEstimatesFile, idxClusterRes->odata);
		printClusterEstimates(vcount.indexClusterFile, idxClusterRes->odata, idxClusterRes->cest);	
	}
						
	if(settings.diClustering || settings.dfiClustering){
		vector<int32_t> boxLabel(colourSel.selectedKeypoints.size(), -1);
		
	#pragma omp parallel for
		for(uint i = 0; i < colourSel.selectedKeypoints.size(); i++){
			KeyPoint kp = colourSel.selectedKeypoints[i];
			vector<int32_t> tmpids;
			for(uint j = 0; j < res1->boxStructures->size(); j++){
				box_structure& bs = res1->boxStructures->at(j);
				if(bs.box.contains(kp.pt)){
					tmpids.push_back(j);
				}
			}
			
			if(!tmpids.empty()){
				double minMoments = res1->boxStructures->at(tmpids[0]).momentsCompare;
				int minIdx = tmpids[0];
				
				for(uint j = 0; j < tmpids.size(); j++){
					double mm = res1->boxStructures->at(tmpids[j]).momentsCompare;
					
					if(mm < minMoments){
						minMoments = mm;
						minIdx = tmpids[j];
					}
				}
				
				boxLabel[i] = minIdx;
			}					
		}
		set<int32_t> lst(boxLabel.begin(), boxLabel.end());
		
		vector<box_structure> bsts;
		Mat fdi = f.frame.clone();
		for(set<int32_t>::iterator it = lst.begin(); it != lst.end(); ++it){
			int32_t bidx = *it;
			if(bidx != -1){
				bsts.push_back(res1->boxStructures->at(bidx));
			}
		}
		
		map<String, Mat> selectedClustersImages;
		selectedClustersImages["img_allkps"] = idxClusterRes->selectedClustersImages->at("img_allkps");
		createBoxStructureImages(&bsts, &selectedClustersImages);
	}		
}

void filteredDescriptorClustering(vocount& vcount, vsettings& settings, selection_t& colourSel, results_t* res1, framed& f, Mat& frame){
	results_t* selDescRes;
	Mat dset = getDescriptorDataset(vcount.frameHistory, settings.step, colourSel.selectedDesc, colourSel.selectedKeypoints, settings.rotationalInvariance, false);	
	selDescRes = do_cluster(NULL, dset, colourSel.selectedKeypoints, 1, 3, true, false);
	generateFinalPointClusters(colourSel.roiFeatures, selDescRes);
	getBoxStructure(selDescRes, f.roi, frame, true, true);	
				
	selDescRes->total = 0; //countPrint(selDescRes->roiClusterPoints, selDescRes->finalPointClusters, 
												//selDescRes->cest, selDescRes->selectedFeatures, selDescRes->lsize);				
	if(settings.print){		
		
		String selectedDescFrameDir = createDirectory(settings.filteredDescDir, to_string(vcount.frameCount));
							
		generateClusterImages(f.frame, selDescRes);
		createBoxStructureImages(selDescRes->boxStructures, selDescRes->selectedClustersImages);
		Mat frm = drawKeyPoints(frame, colourSel.selectedKeypoints, colours.red, -1);
		printImage(settings.filteredDescDir, vcount.frameCount, "frame_kp", frm);
		generateOutputData(vcount, f.frame, colourSel.selectedKeypoints, colourSel.roiFeatures, selDescRes, f.i);
		printImages(selectedDescFrameDir, selDescRes->selectedClustersImages, vcount.frameCount);
		printEstimates(vcount.selDescEstimatesFile, selDescRes->odata);
		printClusterEstimates(vcount.selDescClusterFile, selDescRes->odata, selDescRes->cest);	
	}
				
	f.results[ResultIndex::SelectedKeypoints] = selDescRes;
}

void processFrame(vocount& vcount, vsettings& settings, selection_t& colourSel, Mat& frame){
	
	vcount.frameCount++;
	
	framed f, index_f, sel_f;
	Mat templateMatch;
	
	if(settings.print){
		printImage(settings.outputFolder, vcount.frameCount, "frame", frame);
	}

	f.i = vcount.frameCount;
	index_f.i = f.i;
	sel_f.i = f.i;

	f.frame = frame.clone();
	index_f.frame = f.frame;
	sel_f.frame = f.frame;

	cvtColor(f.frame, f.gray, COLOR_BGR2GRAY);
	vcount.detector->detectAndCompute(frame, Mat(), f.keypoints, f.descriptors);
	if (!f.keypoints.empty()) {
		cout << "################################################################################" << endl;
		cout << "                              " << vcount.frameCount << endl;
		cout << "################################################################################" << endl;
		printf("Frame %d truth is %d\n", vcount.frameCount, vcount.truth[vcount.frameCount]);
		getROI(vcount, settings, f, frame);
		
		display("frame", frame);
        
		f.hasRoi = vcount.roiExtracted;
        results_t* res1 = NULL;
        
        /**
         * Clustering in the descriptor space with unfiltered 
         * dataset. 
         */
		if(settings.dClustering){			
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Original Descriptor Space Clustering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			Mat dset = getDescriptorDataset(vcount.frameHistory, settings.step, f.descriptors, f.keypoints, settings.rotationalInvariance, false);	
			f.results[ResultIndex::Descriptors] = clusterDescriptors(vcount, settings, f, dset, f.keypoints);
		}
		
		/**
		 * Finding the colour model for the current frame
		 * 
		 */ 
		if(colourSel.minPts == -1 && (settings.isClustering || settings.fdClustering)){
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Detecting Colour Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			printf("Finding proper value of minPts\n");
			colourSel = trainColourModel(frame, f.descriptors, f.keypoints, vcount.trainingFile, vcount.trackingFile);	
		} 
		
		if(colourSel.minPts != -1){
			
			if(vcount.frameHistory.size() > 0){
				cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Track Colour Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
				trackFrameColourModel(vcount, f, colourSel, frame);
			}
			
			Mat roiDesc;
			int32_t ce;
			findROIFeature(colourSel.selectedKeypoints, colourSel.selectedDesc, f.roi, colourSel.roiFeatures, roiDesc, ce);			
				
			/****************************************************************************************************/
			/// Image Space Clustering
			/// -------------------------
			/// 
			/****************************************************************************************************/
			if(settings.isClustering){
				cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Image Space Clustering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
				printf("Clustering selected keypoints in image space\n\n\n");
				imageSpaceClustering(vcount, settings, colourSel, f);			
			}				
				
			/****************************************************************************************************/
			/// Selected Colour Model Descriptor Clustering
			/// -------------------------
			/// Create a dataset of descriptors based on the selected colour model
			/// 
			/****************************************************************************************************/
			if(settings.fdClustering || settings.dfClustering || settings.dfiClustering){
				cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Selected Colour Model Descriptor Clustering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
				printf("Clustering selected keypoints in descriptor space\n\n");
				filteredDescriptorClustering(vcount, settings, colourSel, res1, f, frame);	
			}
			
			/****************************************************************************************************/
			/// Filter original descriptor clustering results with the frame colour model
			/****************************************************************************************************/
			if(settings.dfClustering){
				cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Selected Descriptor Space Clustering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
				printf("Filtering detected objects with colour model\n\n");				
				combineSelDescriptorsRawStructures(vcount, f, colourSel, settings.dfComboDir, settings.print);
	
			}
		}		
	}
	maintaintHistory(vcount, f);
}

void combineSelDescriptorsRawStructures(vocount& vcount, framed& f, selection_t& colourSel, String& dfComboDir, bool print){
	vector<int32_t> keypointStructures(colourSel.selectedKeypoints.size(), -1);
	set<uint> selectedStructures;
	results_t* descriptorResults = f.results[ResultIndex::Descriptors];;
	
	/**
	 * For each keypoint in the selected set, we find all box_structures that 
	 * contain the point.
	 */ 
	vector<vector<uint>> filteredStructures(colourSel.selectedKeypoints.size(), vector<uint>());
	for(uint i = 0; i < colourSel.selectedKeypoints.size(); i++){
		vector<uint>& structures = filteredStructures[i];
		KeyPoint kp = colourSel.selectedKeypoints[i];
		for(uint j = 0; j < descriptorResults->boxStructures->size(); j++){
			box_structure& bx = descriptorResults->boxStructures->at(j);
			
			if(bx.box.contains(kp.pt)){
				structures.push_back(j);
			}
		}
	
	}
		
	/**
	 * For those keypoints that are inside multiple structures,
	 * we find out which structure has the smallest moment comparison
	 */ 
	for(uint i = 0; i < keypointStructures.size(); i++){
		vector<uint>& strs = filteredStructures[i];
		
		if(strs.size() > 0){
			uint minIdx = strs[0];
			double minMoment = descriptorResults->boxStructures->at(minIdx).momentsCompare;
			for(uint j = 1; j < strs.size(); j++){
				uint idx = strs[j];
				double moment = descriptorResults->boxStructures->at(idx).momentsCompare;
				
				if(moment < minMoment){
					minIdx = idx;
					minMoment = moment;
				}
			}
			keypointStructures[i] = minIdx;
			selectedStructures.insert(minIdx);
		}
	}
		
	printf("selStructures.size() = %ld\n", selectedStructures.size());

	if(print){
		Mat kimg = drawKeyPoints(f.frame, colourSel.selectedKeypoints, colours.red, -1);
		for(set<uint>::iterator it = selectedStructures.begin(); it != selectedStructures.end(); it++){
			Scalar value;
			
			RNG rng(12345);
			value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
					rng.uniform(0, 255));				
			
			box_structure& b = descriptorResults->boxStructures->at(*it);
			rectangle(kimg, b.box, value, 2, 8, 0);
		}
		printImage(dfComboDir, f.i, "selected_structures", kimg) ;
		double accuracy = 0;
		if(vcount.truth[f.i] > 0){
			accuracy = ((double)selectedStructures.size() / vcount.truth[f.i]) * 100;
		} 
		vcount.dfEstimatesFile << f.i << "," <<  selectedStructures.size() << "," << vcount.truth[f.i] << "," << accuracy << "\n";
	}
}

results_t* clusterDescriptors(vocount& vcount, vsettings& settings, framed& f, Mat& dataset, vector<KeyPoint>& keypoints){	
	
	results_t* res = do_cluster(NULL, dataset, keypoints, settings.step, 3, true, false);
	generateFinalPointClusters(f.roiFeatures, res);	
	getBoxStructure(res, f.roi, f.frame, settings.extend, false);
	//getBoxStructureByHomography(res, f.roi, f.frame, settings.extend, false);
    res->total = 0; //countPrint(res->roiClusterPoints, res->finalPointClusters, res->cest, res->selectedFeatures, res->lsize);
			
	if(settings.print){
		String descriptorFrameDir = createDirectory(settings.descriptorDir, to_string(vcount.frameCount));
		generateClusterImages(f.frame, res);
		createBoxStructureImages(res->boxStructures, res->selectedClustersImages);
		Mat frm = drawKeyPoints(f.frame, keypoints, colours.red, -1);
		printImage(settings.descriptorDir, vcount.frameCount, "frame_kp", frm);					
		generateOutputData(vcount, f.frame, keypoints, f.roiFeatures, res, f.i);
		printImages(descriptorFrameDir, res->selectedClustersImages, vcount.frameCount);
		printEstimates(vcount.descriptorsEstimatesFile, res->odata);
		printClusterEstimates(vcount.descriptorsClusterFile, res->odata, res->cest);	
	}
    	
	return res;
}

void finalise(vocount& vcount){
		
#pragma omp parallel for
	for(uint i = 0; i < vcount.frameHistory.size(); i++){
		framed& f1 = vcount.frameHistory[i];
		for(map<ResultIndex, results_t*>::iterator it = f1.results.begin(); it != f1.results.end(); ++it){
			cleanResult(it->second);
		}
	}

    if(vcount.descriptorsClusterFile.is_open()){
        vcount.descriptorsClusterFile.close();
    }

    if(vcount.descriptorsEstimatesFile.is_open()){
        vcount.descriptorsEstimatesFile.close();
    }

    if(vcount.selDescClusterFile.is_open()){
        vcount.selDescClusterFile.close();
    }

    if(vcount.selDescEstimatesFile.is_open()){
        vcount.selDescEstimatesFile.close();
    }

    if(vcount.indexClusterFile.is_open()){
        vcount.indexClusterFile.close();
    }

    if(vcount.indexEstimatesFile.is_open()){
        vcount.indexEstimatesFile.close();
    }

    if(vcount.trackingFile.is_open()){
        vcount.trackingFile.close();
    }
}
