#include "vocount/voprinter.hpp"
#include <opencv2/imgcodecs.hpp>
#include <QDir>

namespace vocount
{
/**
 *
 */
void VOPrinter::printEstimates(ofstream& myfile, map<OutDataIndex, int32_t>* estimates, double duration)
{
	myfile << estimates->at(OutDataIndex::FrameNum) << ",";
	myfile << estimates->at(OutDataIndex::FeatureSize) << ",";
	myfile << estimates->at(OutDataIndex::SelectedFeatureSize) << ",";
	myfile << estimates->at(OutDataIndex::NumClusters) << ",";
	myfile << estimates->at(OutDataIndex::BoxEst) << ",";
	myfile << estimates->at(OutDataIndex::TruthCount) << ",";
	myfile << estimates->at(OutDataIndex::MinPts) << ",";
	myfile << estimates->at(OutDataIndex::Validity) << ",";
	double accuracy = (estimates->at(OutDataIndex::TruthCount) != 0) ? ((double) estimates->at(OutDataIndex::BoxEst) / estimates->at(OutDataIndex::TruthCount)) * 100 : 0;
	myfile << accuracy << ",";
	myfile << duration << "\n";

	myfile.flush();

}

/**
 *
 */
void VOPrinter::printClusterEstimates(ofstream& myfile, map<OutDataIndex, int32_t>* estimates, vector<int32_t>* cest)
{
	myfile << estimates->at(OutDataIndex::FrameNum) << ",";
	myfile << estimates->at(OutDataIndex::ClusterSum) << ",";
	myfile << estimates->at(OutDataIndex::ClusterAverage) << ",";
	myfile << estimates->at(OutDataIndex::BoxEst) << ",";

	for(vector<int32_t>::iterator it = cest->begin(); it != cest->end(); ++it){

		myfile << *it << ",";
	}
	myfile << "\n";
	myfile.flush();
}

/**
 *
 */
void VOPrinter::printStatistics(map<int32_t, map<String, double>>& stats, String& folder)
{
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

	return dest;
}

/**
 *
 */
String VOPrinter::createDirectory(String& mainFolder, String subfolder)
{
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

	return dest;
}

/**
 *
 */
void VOPrinter::printImages(String& folder, map<String, Mat>* images, int count)
{
	for(map<String, Mat>::iterator it = images->begin(); it != images->end(); ++it){
		printImage(folder, count, it->first, it->second);
	}
}

/**
 *
 */
void VOPrinter::printImage(String folder, int idx, String name, Mat img)
{
	stringstream sstm;
	sstm << folder.c_str() << "/" << idx << "_" << name.c_str() << ".jpg";
	imwrite(sstm.str(), img);
}

/**
 *
 */
void VOPrinter::printMatToFile(const Mat& mtx, String folder, String filename, Formatter::FormatType fmt)
{
	String fname = folder;
	fname += "/";
	fname += filename;

	fname += ".csv";

	if(fmt == Formatter::FMT_MATLAB){
		fname += ".m";
	} else if(fmt == Formatter::FMT_PYTHON){
		fname += ".py";
	} else if(fmt == Formatter::FMT_NUMPY){
		fname += ".py";
	} else if(fmt == Formatter::FMT_C){
		fname += ".h";
	}

	ofstream outfile(fname);
	outfile << format(mtx, fmt) << endl;
	outfile.close();
}

/**
 *
 *
 */
void VOPrinter::printLabelsToFile(int32_t* labels, int32_t length, String folder, Formatter::FormatType fmt)
{
	Mat lbs(length, 1, CV_32S, labels);
	printMatToFile(lbs, folder, "labels", fmt);
}
};
