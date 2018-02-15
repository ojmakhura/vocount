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

void generateFinalPointClusters(vector<vector<int32_t>>& roiFeatures, results_t* res){
	set<int32_t> st;
	for(uint i = 0; i < roiFeatures.size(); i++){
		for (vector<int32_t>::iterator it = roiFeatures[i].begin(); it != roiFeatures[i].end(); ++it) {
			int* key;
			int k = res->labels->at(*it);
			key = &k;
			IntArrayList* list = (IntArrayList *)g_hash_table_lookup(res->roiClusterPoints, key);
			
			if(list == NULL){
				key = (int *)malloc(sizeof(int));
				*key = k;
				list = int_array_list_init_size(roiFeatures[i].size());
				g_hash_table_insert(res->roiClusterPoints, key, list);
				st.insert(k);
			}
					
			int_array_list_append(list, *it);
			
		}
	}
	
	for (set<int32_t>::iterator it = st.begin(); it != st.end(); ++it){
		int32_t key = *it;
		if (key != 0) {
			int_array_list_append(res->objectClusters, key);
		}		
	}
	
	res->objectClusters = hdbscan_sort_by_similarity(res->distancesMap, res->objectClusters, INTRA_DISTANCE_TYPE);
	
	int32_t *data = (int32_t *)res->objectClusters->data;
	for(int32_t i = res->objectClusters->size - 1; i >= 0; i--){		
		IntArrayList* l1 = (IntArrayList*)g_hash_table_lookup(res->clusterMap, data+i);			
		getListKeypoints(*(res->keypoints), l1, (*(res->finalPointClusters))[*(data+i)]);
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
			//cout << stru.box << " \n" << stru.hist << endl;
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
		distance_values *dv = (distance_values *)g_hash_table_lookup(res->distancesMap, &(it->first));
				
		ss += "-";
		ss += to_string((int)dv->cr_confidence);
		ss += "-";
		ss += to_string((int)dv->dr_confidence);
		(*(res->selectedClustersImages))[ss] = kimg;
		kp.insert(kp.end(), it->second.begin(), it->second.end());
	}

	Mat mm = drawKeyPoints(frame, kp, Scalar(0, 0, 255), -1);

	String ss = "img_allkps";
	(*(res->selectedClustersImages))[ss] = mm;
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
void findROIFeature(vector<KeyPoint>& keypoints, Mat& descriptors, vector<Rect2d>& rois, vector<vector<int32_t>>& roiFeatures, vector<Mat>& roiDesc, vector<int32_t>& centerFeatures){
	roiFeatures.reserve(rois.size());
	roiDesc.reserve(rois.size());
	centerFeatures.reserve(rois.size());
	printf("rois.size = %lu roiFeatures = %lu roiDesc = %lu\n", rois.size(), roiFeatures.size(), roiDesc.size());
//#pragma omp parallel for	
	for(uint i = 0; i < rois.size(); i++){
		roiFeatures.push_back(vector<int32_t>());
		roiDesc.push_back(Mat());
		centerFeatures.push_back(-1);
	}
	printf("rois.size = %lu roiFeatures = %lu roiDesc = %lu\n", rois.size(), roiFeatures.size(), roiDesc.size());

	printf("keypoints = %lu, descriptors = %d\n", keypoints.size(), descriptors.rows);

//#pragma omp parallel for	
	for(uint x = 0; x < rois.size(); x++){
		Rect2d& r = rois[x];

		Point2f p;

		p.x = (r.x + r.width)/2.0f;
		p.y = (r.y + r.height)/2.0f;
		double distance;
		int32_t centerFeature = -1;
		for(uint i = 0; i < keypoints.size(); ++i){
			
			
			for(uint j = 0; j < rois.size(); j++){		
			
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
					Mat t = descriptors.row(i);
					roiDesc[j].push_back(t);
					
				}
			}
		}
		centerFeatures[x] = centerFeature;
		//break;
	}
	//printf("roiDesc had %d rows\n", roiDesc.rows);
	//return centerFeature;
}

void getBoxStructure(results_t* res, vector<Rect2d>& rois, Mat& frame, bool extend){
	vector<vector<box_structure>> b_structures;
	set<int32_t> processedClusters;
	int32_t *data = (int32_t *)res->objectClusters->data;
	
	for(int32_t i = res->objectClusters->size - 1; i >= 0; i--){
		int32_t key = data[i];
		processedClusters.insert(key);
		IntArrayList *roiPoints = (IntArrayList *)g_hash_table_lookup(res->roiClusterPoints, &(key));
		vector<vector<KeyPoint>> kps;
		if(roiPoints->size > 1){
			IntArrayList *cPoints = (IntArrayList *)g_hash_table_lookup(res->clusterMap, &(key));
			splitROICluster(roiPoints, cPoints, *(res->dataset), kps, *(res->keypoints));
		} else{
			kps.push_back(res->finalPointClusters->at(key));
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
			addToBoxStructure(&str2, res->finalPointClusters->at(key), kp, mbs, frame);
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
		//extendBoxClusters(frame, res, processedClusters);
	}
}

/**
 * Given a vector of box structures, the function draws the rectangles around the identified object locations
 * 
 */ 
void createBoxStructureImages(vector<box_structure>* boxStructures, map<String, Mat>* selectedClustersImages){
	
	printf("boxStructure found %lu objects\n\n", boxStructures->size());
	String ss = "img_bounds";

	Mat img_bounds = (*selectedClustersImages)["img_allkps"].clone();
	for (size_t i = 0; i < boxStructures->size(); i++) {
		box_structure b = (*boxStructures)[i];
		RNG rng(12345);
		Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
				rng.uniform(0, 255));
		rectangle(img_bounds, b.box, value, 2, 8, 0);
		//Point center = (b.box.br() + b.box.tl())/2;
		//circle(img_bounds, center, 4, Scalar(255, 255, 255), CV_FILLED, 10, 0);
	}
	(*selectedClustersImages)[ss] = img_bounds;
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
	res->labels = new vector<int32_t>(res->keypoints->size());
	res->boxStructures = new vector<box_structure>();
	res->cest = new vector<int32_t>();
	res->selectedClustersImages = new map<String, Mat>();
	res->leftoverClusterImages = new map<String, Mat>();
	res->objectClusters = int_array_list_init();
		
    res->clusterMap = NULL;		 								/// maps labels to the keypoint indices
    res->roiClusterPoints = g_hash_table_new(g_int_hash, g_int_equal);								/// cluster labels for the region of interest mapped to the roi points in the cluster
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
		
		//if(res->stats != NULL){
		//	hdbscan_destroy_stats_map(res->stats);
		//}
		
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
			hdbscan_calculate_stats(res->distancesMap, &(res->stats));
			res->validity = hdbscan_analyse_stats(&(res->stats));
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
		
		/*if(res->stats != NULL){
			hdbscan_destroy_stats_map(res->stats);
			res->stats = NULL;
		}*/
		
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
		
		int_array_list_delete(res->objectClusters);
		
		delete res->dataset;
		delete res->keypoints;
		delete res->finalPointClusters;
		delete res->odata;
		delete res->labels;
		delete res->boxStructures;
		delete res->cest;
		delete res->selectedClustersImages;
		delete res->leftoverClusterImages;
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

void processFrame(vocount& vcount, vsettings& settings, selection_t& colourSel, Mat& frame){
	
	vcount.frameCount++;
	framed f, index_f, sel_f;

	f.i = vcount.frameCount;
	index_f.i = f.i;
	sel_f.i = f.i;

	f.frame = frame.clone();
	index_f.frame = f.frame;
	sel_f.frame = f.frame;

	cvtColor(f.frame, f.gray, COLOR_BGR2GRAY);
	vcount.detector->detectAndCompute(frame, Mat(), f.keypoints, f.descriptors);
	
	if (settings.selectROI && !vcount.roiExtracted) { // select a roi if c has been pressed or if the program was run with -s option
			
		Mat f2 = frame.clone();
		Rect2d boundingBox = selectROI("Select ROI", f2);
		destroyWindow("Select ROI");
		f.rois.push_back(boundingBox);
		cout << boundingBox << endl;
			
		for(size_t i = 0; i < f.rois.size(); i++){
			vcount.trackers.push_back(createTrackerByName(settings.trackerAlgorithm));
			vcount.trackers[i]->init( frame, f.rois[i] );
		}
			
		//trackers.add(_trackers, f2, f.rois);
		vcount.roiExtracted = true;
		settings.selectROI = false;
	}

	if (vcount.roiExtracted ){
		f.rois.resize(1);
		//printf("rois size = %lu\n", f.rois.size());
		RNG rng(12345);
		Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),	rng.uniform(0, 255));
		for(size_t i = 0; i < vcount.trackers.size(); i++){
			vcount.trackers[i]->update(frame, f.rois[i]);
			rectangle(frame, f.rois[i], value, 2, 8, 0);
			break;
		}
	}
	
    //String colourFrameDir;
   	String iSpaceFrameDir;
   	String descriptorFrameDir;
    String selectedDescFrameDir;
    	
    if(settings.print){
		printImage(settings.outputFolder, vcount.frameCount, "frame", frame);
    	//colourFrameDir = createDirectory(settings.colourDir, to_string(vcount.frameCount));
    	iSpaceFrameDir = createDirectory(settings.imageSpaceDir, to_string(vcount.frameCount));
    	descriptorFrameDir = createDirectory(settings.descriptorDir, to_string(vcount.frameCount));
    	selectedDescFrameDir = createDirectory(settings.filteredDescDir, to_string(vcount.frameCount));
    }
        
	if (!f.descriptors.empty()) {
		vector<Rect2d> foundRects;
		cout << "################################################################################" << endl;
		cout << "                              " << f.i << endl;
		cout << "################################################################################" << endl;
		
		// Create clustering dataset
		f.hasRoi = vcount.roiExtracted;
		results_t* res1;
		if(settings.dClustering || settings.diClustering || settings.dfClustering || settings.dfiClustering){
			res1 = clusterDescriptors(vcount, settings, f, iSpaceFrameDir, settings.imageSpaceDir);
		}
		
		if(vcount.frameHistory.size() > 0 && (settings.isClustering || settings.fdClustering || settings.diClustering || settings.dfClustering || settings.dfiClustering)){
			
			if(colourSel.minPts == -1){
				printf("Finding proper value of minPts\n");
				colourSel = detectColourSelectionMinPts(frame, f.descriptors, f.keypoints);
				//Mat kimg = drawKeyPoints(f.frame, colourSel.selectedKeypoints, Scalar(0, 0, 255), -1);
				//display("Selected Keypoints 11", kimg);
				//printf("Finding value of minPts = %d with colourSel.selectedPts as %d from %lu\n", colourSel.minPts, colourSel.selectedDesc.rows, f.keypoints.size());
			
			} else {
				framed ff = vcount.frameHistory[vcount.frameHistory.size()-1];
				vector<KeyPoint> keyp(ff.keypoints.begin(), ff.keypoints.end());
				Mat dataset = getColourDataset(ff.frame, keyp);
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
				colourSel.clusterKeypointIdx = hdbscan_create_cluster_table(scanis.clusterLabels + ff.keypoints.size(), 0, f.keypoints.size());
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
					getListKeypoints(f.keypoints, list, colourSel.selectedKeypoints);
					getSelectedKeypointsDescriptors(f.descriptors, list, selDesc);
				}					
				colourSel.selectedDesc = selDesc.clone();
				
				vector<Mat> roiDesc;
				vector<int32_t> ce;
				findROIFeature(colourSel.selectedKeypoints, colourSel.selectedDesc, f.rois, colourSel.roiFeatures, roiDesc, ce);			
					
				if(settings.isClustering || settings.diClustering || settings.dfiClustering){
					printf("Clustering selected keypoints in image space\n\n\n");
					Mat ds = getImageSpaceDataset(colourSel.selectedKeypoints);
					results_t* idxClusterRes = do_cluster(NULL, ds, colourSel.selectedKeypoints, 1, 3, true, true);
					set<int> ss(idxClusterRes->labels->begin(), idxClusterRes->labels->end());
					printf("We found %lu objects by index points clustering.\n", ss.size() - 1);
					getKeypointMap(idxClusterRes->clusterMap, &colourSel.selectedKeypoints, *(idxClusterRes->finalPointClusters));
					generateClusterImages(f.frame, idxClusterRes);
					f.results["im_space"] = idxClusterRes;		
					
					if(settings.print){
						generateOutputData(vcount, f.frame, colourSel.selectedKeypoints, colourSel.roiFeatures, idxClusterRes, f.i);
						Mat frm = drawKeyPoints(f.frame, colourSel.selectedKeypoints, Scalar(0, 0, 255), -1);
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
									//boxLabel[i] = j;
									//break;
								}
							}
							//printf("tmpids has %lu size\n", tmpids.size());
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
						//int cou = 0;
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
						String bidest = "/mnt/2TB/programming/phd/workspace/_vocount/out/wx02/di";
						printImages(bidest, &selectedClustersImages, vcount.frameCount);
						printf("Combination di found %lu objects\n", lst.size());
					}						
				}				
				
				/****************************************************************************************************/
				/// Selected Colour Model Descriptor Clustering
				/// -------------------------
				/// Create a dataset of descriptors based on the selected colour model
				/// 
				/****************************************************************************************************/
				if(settings.fdClustering || settings.dfClustering || settings.dfiClustering){
					//dataset = colourSel.selectedDesc.clone();
					printf("Clustering selected keypoints in descriptor space\n\n\n");
					results_t* selDescRes = do_cluster(NULL, colourSel.selectedDesc, colourSel.selectedKeypoints, 1, 3, true, false);
					generateFinalPointClusters(colourSel.roiFeatures, selDescRes);
					getBoxStructure(selDescRes, f.rois, frame, false);								
					generateClusterImages(f.frame, selDescRes);
					createBoxStructureImages(selDescRes->boxStructures, selDescRes->selectedClustersImages);
					
					cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
					selDescRes->total = countPrint(selDescRes->roiClusterPoints, selDescRes->finalPointClusters, 
													selDescRes->cest, selDescRes->selectedFeatures, selDescRes->lsize);
					cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
												
						
					if(settings.print){
						Mat frm = drawKeyPoints(frame, colourSel.selectedKeypoints, Scalar(0, 0, 255), -1);
						printImage(settings.filteredDescDir, vcount.frameCount, "frame_kp", frm);
						generateOutputData(vcount, f.frame, colourSel.selectedKeypoints, colourSel.roiFeatures, selDescRes, f.i);
						printImages(selectedDescFrameDir, selDescRes->selectedClustersImages, vcount.frameCount);
						printEstimates(vcount.selDescEstimatesFile, selDescRes->odata);
						printClusterEstimates(vcount.selDescClusterFile, selDescRes->odata, selDescRes->cest);	
					}
					
					f.results["sel_keypoints"] = selDescRes;
				}
			}
		}		
	}
	maintaintHistory(vcount, f);
}

results_t* clusterDescriptors(vocount& vcount, vsettings& settings, framed& f, String& keypointsFrameDir, String& keypointsDir){
	
	findROIFeature(f.keypoints, f.descriptors, f.rois, f.roiFeatures, f.roiDesc, f.centerFeatures);
	Mat dset = getDescriptorDataset(vcount.frameHistory, settings.step, f.descriptors);
	results_t* res = do_cluster(NULL, f.descriptors, f.keypoints, settings.step, 3, true, true);
	generateFinalPointClusters(f.roiFeatures, res);	
	getBoxStructure(res, f.rois, f.frame, true);
	generateClusterImages(f.frame, res);
	createBoxStructureImages(res->boxStructures, res->selectedClustersImages);
	//printf("Frame %d truth is %d\n", vcount.frameCount, vcount.truth[vcount.frameCount]);
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	res->total = countPrint(res->roiClusterPoints, res->finalPointClusters, res->cest, res->selectedFeatures, res->lsize);
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
	cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;

	f.results["descriptors"] = res;
				
	if(settings.print){
		Mat frm = drawKeyPoints(f.frame, f.keypoints, Scalar(0, 0, 255), -1);
		printImage(keypointsDir, vcount.frameCount, "frame_kp", frm);					
		generateOutputData(vcount, f.frame, f.keypoints, f.roiFeatures, res, f.i);
		printImages(keypointsFrameDir, res->selectedClustersImages, vcount.frameCount);
		printEstimates(vcount.descriptorsEstimatesFile, res->odata);
		printClusterEstimates(vcount.descriptorsClusterFile, res->odata, res->cest);	
	}
	/*
	if(res1->validity == 4){
		vector<Ptr<Tracker>> ts;
		findNewROIs(frame, trackers, f.rois, res->boxStructures, vcount.trackerAlgorithm);
		printf("(foundRects, res->boxStructures) = (%lu, %lu)\n", foundRects.size(), res->boxStructures->size());
	}*/
	
	return res;
}

void finalise(vocount& vcount){
		
#pragma omp parallel for
	for(uint i = 0; i < vcount.frameHistory.size(); i++){
		framed& f1 = vcount.frameHistory[i];
		for(map<String, results_t*>::iterator it = f1.results.begin(); it != f1.results.end(); ++it){
			cleanResult(it->second);
		}
	}

    //if(vcount.print){
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
    //}
}
