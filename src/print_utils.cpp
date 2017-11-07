/*
 * print_utils.cpp
 *
 *  Created on: 1 Aug 2017
 *      Author: ojmakh
 */
#include "vocount/print_utils.hpp"

void printImage(String folder, int idx, String name, Mat img) {
	stringstream sstm;

	sstm << folder.c_str() << "/" << idx << " " << name.c_str() << ".jpg";
	imwrite(sstm.str(), img);
}


void printEstimates(ofstream& myfile, map<String, int32_t>* estimates){
	myfile << estimates->at(frameNum) << ",";
	myfile << estimates->at(sampleSize) << ",";
	myfile << estimates->at(selectedSampleSize) << ",";
	myfile << estimates->at(featureSize) << ",";
	myfile << estimates->at(selectedFeatureSize) << ",";
	myfile << estimates->at(numClusters) << ",";
	myfile << estimates->at(clusterSum) << ",";
	myfile << estimates->at(clusterAverage) << ",";
	myfile << estimates->at(boxEst) << ",";
	myfile << estimates->at(truthCount) << "\n";
	
	myfile.flush();

}

void printClusterEstimates(ofstream& myfile, map<String, int32_t>* estimates, vector<int32_t>* cest){
	
	myfile << estimates->at(frameNum) << ",";
	myfile << estimates->at(clusterSum) << ",";
	myfile << estimates->at(clusterAverage) << ",";
	myfile << estimates->at(boxEst) << ",";
	
	for(vector<int32_t>::iterator it = cest->begin(); it != cest->end(); ++it){
		
		myfile << *it << ",";
	}
	myfile << "\n";
	myfile.flush();

}

void printStatistics(map<int32_t, map<String, double>>& stats, String folder){
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

	for(map<int32_t, map<String, double>>::iterator it = stats.begin(); it != stats.end(); ++it){

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

void generateOutputData(vocount& vcount, Mat& frame, vector<KeyPoint>& keypoints, vector<int>& roiFeatures, results_t* res, int i){
	if (vcount.print && g_hash_table_size(res->roiClusterPoints) > 0) {

		printImage(vcount.destFolder, vcount.frameCount, "frame", frame);

		Mat ff = drawKeyPoints(frame, keypoints, Scalar(0, 0, 255), -1);
		printImage(vcount.destFolder, vcount.frameCount, "frame_kp", ff);

		(*(res->odata))[sampleSize] = roiFeatures.size();

		int selSampleSize = 0;

		GHashTableIter iter;
		gpointer key;
		gpointer value;
		g_hash_table_iter_init (&iter, res->roiClusterPoints);
		
		while (g_hash_table_iter_next (&iter, &key, &value)){
			IntArrayList* list = (IntArrayList*)value;
			selSampleSize += list->size;
		}

		(*(res->odata))[selectedSampleSize] = selSampleSize;
		(*(res->odata))[featureSize] = res->ogsize;
		(*(res->odata))[selectedFeatureSize] = res->selectedFeatures;
		(*(res->odata))[numClusters] = res->keyPointImages->size();
		(*(res->odata))[clusterSum] = res->total;
		int32_t avg = res->total / res->keyPointImages->size();
		(*(res->odata))[clusterAverage] = avg;
		(*(res->odata))[boxEst] = res->boxStructures->size();
		(*(res->odata))[frameNum] = i;

		if(i >= vcount.truth.size()){
			(*(res->odata))[truthCount] = 0;
		} else{
			(*(res->odata))[truthCount] = vcount.truth[i-1];
		}
	}
}

