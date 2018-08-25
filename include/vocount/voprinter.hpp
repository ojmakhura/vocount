#ifndef VOPRINTER_H_
#define VOPRINTER_H_

#include "vocount_types.hpp"
#include <QDir>

namespace vocount {
class VOPrinter
{
public:

	/**
	 *
	 */
	static void printEstimates(ofstream& myfile, map<OutDataIndex, int32_t>* estimates);

	/**
	 *
	 */
	static void printClusterEstimates(ofstream& myfile, map<OutDataIndex, int32_t>* estimates, vector<int32_t>* cest);

	/**
	 *
	 */
	static void printStatistics(map<int32_t, map<String, double>>& stats, String& folder);

	/**
	 *
	 */
	static String createDirectory(String& mainFolder, String subfolder);

	/**
	 *
	 */
	static void printImages(String& folder, map<String, UMat>* images, int count);

	/**
	 *
	 */
	static void printImage(String folder, int idx, String name, UMat img);

	/**
	 *
	 */
	static void printMatToFile(const UMat& mtx, String folder, String filename, int fmt);

	/**
	 *
	 *
	 */
	static void printLabelsToFile(int32_t* labels, int32_t length, String folder, int fmt);
};
};
#endif // VOPRINTER_H_
