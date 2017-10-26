/*
 * print_utils.cpp
 *
 *  Created on: 1 Aug 2017
 *      Author: ojmakh
 */
#include "vocount/print_utils.hpp"
#include <fstream>

void printImage(String folder, int idx, String name, Mat img) {
	stringstream sstm;

	sstm << folder.c_str() << "/" << idx << " " << name.c_str() << ".jpg";
	imwrite(sstm.str(), img);
}


void printStats(String folder, map<int32_t, vector<int32_t> > stats){
	ofstream myfile;
	String f = folder;
	String name = "/stats.csv";
	f += name;
	myfile.open(f.c_str());

	myfile << "Frame #,Sample Size,Selected Sample,Feature Size, Selected Features, # Clusters,Cluster Sum, Cluster Avg., Box Est.,Actual\n";

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

	myfile << "Frame #,Cluster Sum, Cluster Avg., Box Est.\n";

	for(map<int32_t, vector<int32_t> >::iterator it = cEstimates.begin(); it != cEstimates.end(); ++it){
		vector<int32_t> vv = it->second;
		size_t sz = vv.size();
		myfile << it->first << "," << vv[sz-1] << "," << vv[sz-2] << "," << vv[sz-3] << ",";

		for(uint i = 0; i < vv.size()-2; ++i){
			myfile << vv[i] << ",";
		}
		myfile << "\n";
	}

	myfile.close();

}

void printStatistics(map<int, map<String, double>> stats, String folder){
	printf("Printing statistics to %s.\n", folder.c_str());
	ofstream coreFile, disFile;
	String name = "/core_distance_statistics.csv";
	String f = folder;
	f += name;
	coreFile.open(f.c_str());

	f = folder;
	name = "/distance_statistics.csv";
	f += name;
	disFile.open(f.c_str());

	coreFile << "minPts, Mean, Variance, Standard Deviation, Kurtosis, Skewness, Count\n";
	disFile << "minPts, Mean, Variance, Standard Deviation, Kurtosis, Skewness, Count\n";

	for(map<int, map<String, double>>::iterator it = stats.begin(); it != stats.end(); ++it){

		map<String, double> mp = it->second;
		coreFile << it->first << ",";
		coreFile << mp["mean_cr"] << ",";
		coreFile << mp["variance_cr"] << ",";
		coreFile << mp["sd_cr"] << ",";
		if(mp["kurtosis_cr"] == std::numeric_limits<double>::max()){
			coreFile << "NaN" << ",";
		} else {
			coreFile << mp["kurtosis_cr"] << ",";
		}

		if(mp["skew_cr"] == std::numeric_limits<double>::max()){
			coreFile << "NaN" << ",";
		} else {
			coreFile << mp["skew_cr"] << ",";
		}
		coreFile << mp["count"] << "\n";

		disFile << it->first << ",";
		disFile << mp["mean_dr"] << ",";
		disFile << mp["variance_dr"] << ",";
		disFile << mp["sd_dr"] << ",";
		if(mp["kurtosis_dr"] == std::numeric_limits<double>::max()){
			disFile << "NaN" << ",";
		} else {
			disFile << mp["kurtosis_dr"] << ",";
		}

		if(mp["skew_dr"] == std::numeric_limits<double>::max()){
			disFile << "NaN" << ",";
		} else {
			disFile << mp["skew_dr"] << ",";
		}
		disFile << mp["count"] << "\n";

	}

	coreFile.close();
	disFile.close();

}

String createDirectory(String& mainFolder, String subfolder){
	String sokp = mainFolder;
	sokp += "/";
	sokp += subfolder;

	String command = "mkdir \'";
	command += sokp;
	command += "\'";
	printf(command.c_str());
	printf("\n");
	const int dir_err2 = system(command.c_str());
	if (-1 == dir_err2) {
		printf("Error creating directory!n");
		exit(1);
	}

	return sokp;

}

void printImages(String folder, map<String, Mat>* images, int count){
	for(map<String, Mat>::iterator it = images->begin(); it != images->end(); ++it){
		printImage(folder, count, it->first, it->second);
	}
}

void printData(vocount& vcount, Mat& frame, vector<KeyPoint>& keypoints, vector<int>& roiFeatures, results_t& res, int i){
	if (vcount.print && g_hash_table_size(res.roiClusterPoints) > 0) {

		printImage(vcount.destFolder, vcount.frameCount, "frame", frame);

		Mat ff = drawKeyPoints(frame, keypoints, Scalar(0, 0, 255), -1);
		printImage(vcount.destFolder, vcount.frameCount, "frame_kp", ff);

		res.odata->push_back(roiFeatures.size());

		int selSampleSize = 0;

		GHashTableIter iter;
		gpointer key;
		gpointer value;
		g_hash_table_iter_init (&iter, res.roiClusterPoints);
		
		while (g_hash_table_iter_next (&iter, &key, &value)){
			IntArrayList* list = (IntArrayList*)value;
			selSampleSize += list->size;
			
		}

		res.odata->push_back(selSampleSize);
		res.odata->push_back(res.ogsize);
		res.odata->push_back(res.selectedFeatures);
		res.odata->push_back(res.keyPointImages->size());
		res.odata->push_back(res.total);
		int32_t avg = res.total / res.keyPointImages->size();
		res.odata->push_back(avg);
		res.odata->push_back(res.boxStructures->size());

		map<int, int>::iterator it = vcount.truth.find(i);

		if(it == vcount.truth.end()){
			res.odata->push_back(0);
		} else{
			res.odata->push_back(it->second);
		}
		pair<int32_t, vector<int32_t> > pp(vcount.frameCount, *res.odata);
		vcount.stats.insert(pp);
		res.cest->push_back(res.boxStructures->size());
		res.cest->push_back(avg);
		res.cest->push_back(res.total);
		pair<int32_t, vector<int32_t> > cpp(vcount.frameCount, *res.cest);
		vcount.clusterEstimates.insert(cpp);
	}
}

