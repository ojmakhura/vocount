#ifndef VOCUTILS_H
#define VOCUTILS_H

#include "vocount_types.hpp"
#include <listlib/intlist.h>

namespace vocount
{

class VOCUtils
{
public:
	/**
	 *
	 */
    static set<int32_t> findValidROIFeature(vector<KeyPoint>& keypoints, Rect2d& roi, vector<int32_t>& roiFeatures, vector<int32_t>& labels);

	/**
	 *
	 */
    static void findROIFeatures(vector<KeyPoint>& keypoints, Rect2d& roi, vector<int32_t>& roiFeatures);

	/**
	 *
	 */
    static void sortByDistanceFromCenter(Rect2d& roi, vector<int32_t>& roiFeatures, vector<KeyPoint>& keypoints);

	/**
	 *
	 */
    static UMat calculateHistogram(UMat img_);

	/**
	 *
	 */
    static cv::Ptr<cv::Tracker> createTrackerByName(cv::String name);

	/**
	 *
	 */
    static bool trimRect(Rect2d& r, int32_t rows, int32_t cols, int32_t padding);

	/**
	 *
	 */
    static bool stabiliseRect(UMat frame, Rect2d templ_r, Rect2d& proposed);

	/**
	 *
	 */
    static bool _stabiliseRect(UMat frame, Rect2d templ_r, Rect2d& proposed);

	/**
	 *
	 */
	static Rect2d shiftRect(Rect2d box, Point2f first, Point2f second);

	/**
	 *
	 */
	static void getListKeypoints(vector<KeyPoint>& keypoints, IntArrayList* list, vector<KeyPoint>& out);

	/**
	 *
	 */
	static void getVectorKeypoints(vector<KeyPoint>& keypoints, vector<int32_t>& list, vector<KeyPoint>& out);

	/**
	 *
	 */
	static UMat drawKeyPoints(UMat in, vector<KeyPoint>& points, Scalar colour, int32_t type);

	/**
	 *
	 */
	static UMat getDescriptorDataset(UMat descriptors, vector<KeyPoint>& keypoints, bool includeAngle, bool includeOctave);

	/**
	 *
	 */
	static void display(char const* screen, const InputArray& m);

	/**
	 *
	 */
	static UMat getColourDataset(UMat f, vector<KeyPoint>& pts);

	/**
	 *
	 */
	static void quickSortByDistance(vector<int32_t>& roiFeatures, vector<double>& distances, int low, int high);

	/**
	 *
	 */
	static double calcDistanceL1(Point2f f1, Point2f f2);
};
}
#endif // VOCUTILS_H
