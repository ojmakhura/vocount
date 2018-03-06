/*
 * print_utils.cpp
 *
 *  Created on: 1 Aug 2017
 *      Author: ojmakh
 */
#include "vocount/print_utils.hpp"
#include <QDir>

void createOutputDirectories(vocount& vcount, vsettings& settings){
	
	if(settings.print){
		createDirectory(settings.outputFolder, "");
		printf("Created %s directory.\n", settings.outputFolder.c_str());
		
		if(settings.dClustering){
			settings.descriptorDir = createDirectory(settings.outputFolder, "descriptors");
			printf("Created descriptors directory at %s.\n", settings.descriptorDir.c_str());
			
			String name = settings.descriptorDir + "/estimates.csv";
			vcount.descriptorsEstimatesFile.open(name.c_str());
			vcount.descriptorsEstimatesFile << "Frame #,Sample Size,Selected Sample,Feature Size, Selected Features, # Clusters,Cluster Sum, Cluster Avg., Box Est.,Actual, Validity, Accuracy\n";
				
			name = settings.descriptorDir + "/ClusterEstimates.csv";
			vcount.descriptorsClusterFile.open(name.c_str());
			vcount.descriptorsClusterFile << "Frame #,Cluster Sum, Cluster Avg., Box Est.\n";
		}
		
		if(settings.isClustering){
			settings.imageSpaceDir = createDirectory(settings.outputFolder, "image_space");
			printf("Created image_space directory at %s.\n", settings.imageSpaceDir.c_str());
			
			String name = settings.imageSpaceDir + "/estimates.csv";
			vcount.indexEstimatesFile.open(name.c_str());
			vcount.indexEstimatesFile << "Frame #,Sample Size,Selected Sample,Feature Size, Selected Features, # Clusters,Cluster Sum, Cluster Avg., Box Est.,Actual, Validity, Accuracy\n";
			
			name = settings.imageSpaceDir + "/ClusterEstimates.csv";
			vcount.indexClusterFile.open(name.c_str());
			vcount.indexClusterFile << "Frame #,Cluster Sum, Cluster Avg., Box Est.\n";
		}
		
		if(settings.fdClustering){
			settings.filteredDescDir = createDirectory(settings.outputFolder, "filtered_descriptors");
			printf("Created filtered_descriptors directory at %s.\n", settings.filteredDescDir.c_str());
			
			String name = settings.filteredDescDir + "/estimates.csv";
			vcount.selDescEstimatesFile.open(name.c_str());
			vcount.selDescEstimatesFile << "Frame #,Sample Size,Selected Sample,Feature Size, Selected Features, # Clusters,Cluster Sum, Cluster Avg., Box Est.,Actual, Validity, Accuracy\n";
			
			name = settings.filteredDescDir + "/ClusterEstimates.csv";
			vcount.selDescClusterFile.open(name.c_str());
			vcount.selDescClusterFile << "Frame #,Cluster Sum, Cluster Avg., Box Est.\n";
		}
		
		if(settings.dfClustering){
			settings.dfComboDir = createDirectory(settings.outputFolder, "descriptor_filtered_descriptors");
			printf("Created descriptor_filtered_descriptors directory at %s.\n", settings.dfComboDir.c_str());
			
			String name = settings.dfComboDir + "/estimates.csv";
			vcount.dfEstimatesFile.open(name.c_str());
			vcount.dfEstimatesFile << "Frame #,Box Est.,Actual, Accuracy\n";
				
			//name = settings.dfComboDir + "/ClusterEstimates.csv";
			//vcount.dfClusterFile.open(name.c_str());
			//vcount.dfClusterFile << "Frame #,Cluster Sum, Cluster Avg., Box Est.\n";
		}
		
		if(settings.diClustering){
			settings.diComboDir = createDirectory(settings.outputFolder, "descriptor_image_space");
			printf("Created filtered_descriptors directory at %s.\n", settings.diComboDir.c_str());
			
			String name = settings.diComboDir + "/estimates.csv";
			vcount.diEstimatesFile.open(name.c_str());
			vcount.diEstimatesFile << "Frame #,Sample Size,Selected Sample,Feature Size, Selected Features, # Clusters,Cluster Sum, Cluster Avg., Box Est.,Actual, Validity, Accuracy\n";
				
			name = settings.diComboDir + "/ClusterEstimates.csv";
			vcount.diClusterFile.open(name.c_str());
			vcount.diClusterFile << "Frame #,Cluster Sum, Cluster Avg., Box Est.\n";
		}
		
		if(settings.dfiClustering){
			settings.dfiComboDir = createDirectory(settings.outputFolder, "all_combined");
			printf("Created all_combined directory at %s.\n", settings.dfiComboDir.c_str());
			
			String name = settings.dfiComboDir + "/estimates.csv";
			vcount.dfiEstimatesFile.open(name.c_str());
			vcount.dfiEstimatesFile << "Frame #,Sample Size,Selected Sample,Feature Size, Selected Features, # Clusters,Cluster Sum, Cluster Avg., Box Est.,Actual, Validity, Accuracy\n";
				
			name = settings.dfiComboDir + "/ClusterEstimates.csv";
			vcount.dfiClusterFile.open(name.c_str());
			vcount.dfiClusterFile << "Frame #,Cluster Sum, Cluster Avg., Box Est.\n";
		}
	}
	
}

void printImage(String folder, int idx, String name, Mat img) {
	stringstream sstm;

	sstm << folder.c_str() << "/" << idx << " " << name.c_str() << ".jpg";
	//cout << "Trying to print %s\n" << sstm.str() << endl;
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
	myfile << estimates->at(truthCount) << ",";
	myfile << estimates->at(validityStr) << ",";
	double accuracy = estimates->at(truthCount) != 0 ? ((double estimates->at(boxEst)) / estimates->at(truthCount)) * 100 : 0;
	myfile << accuracy << "\n";
	
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

void printStatistics(map<int32_t, map<String, double>>& stats, String& folder){
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
	String dest = mainFolder;
	dest += "/";
	dest += subfolder;
	
	QDir main(mainFolder.c_str());
	
	if(!QDir(dest.c_str()).exists()){
		if(!QDir().mkpath(dest.c_str())){
			
			printf("Error creating directory %s\n", dest.c_str());
			exit(1);
		}
	}

	/*String command = "mkdir \'";
	command += sokp;
	command += "\'";
	printf("%s\n", command.c_str());
	printf("\n");
	const int dir_err2 = system(command.c_str());
	if (-1 == dir_err2) {
		printf("Error creating directory!n");
		exit(1);
	}*/

	return dest;

}

void printImages(String& folder, map<String, Mat>* images, int count){
	for(map<String, Mat>::iterator it = images->begin(); it != images->end(); ++it){
		printImage(folder, count, it->first, it->second);
	}
}

void generateOutputData(vocount& vcount, Mat& frame, vector<KeyPoint>& keypoints, vector<vector<int32_t>>& roiFeatures, results_t* res, int i){
	//if (vcount.print) {
		int selSampleSize = 0;
		if(g_hash_table_size(res->roiClusterPoints) > 0){
			(*(res->odata))[sampleSize] = roiFeatures[0].size();
			GHashTableIter iter;
			gpointer key;
			gpointer value;
			g_hash_table_iter_init (&iter, res->roiClusterPoints);
			
			while (g_hash_table_iter_next (&iter, &key, &value)){
				IntArrayList* list = (IntArrayList*)value;
				selSampleSize += list->size;
			}
			(*(res->odata))[boxEst] = res->boxStructures->size();
		} else{
			(*(res->odata))[sampleSize] = 0;
			res->total = 0;
			(*(res->odata))[boxEst] = g_hash_table_size(res->clusterMap) - 1;
		}

		(*(res->odata))[selectedSampleSize] = selSampleSize;
		(*(res->odata))[featureSize] = res->ogsize;
		(*(res->odata))[selectedFeatureSize] = res->selectedFeatures;
		(*(res->odata))[numClusters] = res->selectedClustersImages->size();
		(*(res->odata))[clusterSum] = res->total;
		int32_t avg = res->total / res->selectedClustersImages->size();
		(*(res->odata))[clusterAverage] = avg;
		(*(res->odata))[frameNum] = i;
		(*(res->odata))[validityStr] = res->validity;

		if((size_t)i > vcount.truth.size()){
			(*(res->odata))[truthCount] = 0;
		} else{
			(*(res->odata))[truthCount] = vcount.truth[i];
		}
	//}
}

