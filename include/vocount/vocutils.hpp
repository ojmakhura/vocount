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
	 * Find the valid feature. This function uses the labels to validate that the
	 * selected features are not part of the noise clusters.
	 * @param keypoints		- local feature keypoints
	 * @param roi			- region of interest
	 * @param roiFeatures	- valid roiFeatures output
	 * @param labels		- Clusters labels
	 */
    static set<int32_t> findValidROIFeatures(vector<KeyPoint>* keypoints, Rect2d& roi, vector<int32_t>* roiFeatures, vector<int32_t>* labels);

	/**
	 * Find all keypoints inside the roi
	 *
	 * @param keypoints		- local feature keypoints
	 * @param roi			- region of interest
	 * @param roiFeatures	- valid roiFeatures output
	 */
    static void findROIFeatures(vector<KeyPoint>* keypoints, Rect2d& roi, vector<int32_t>* roiFeatures);

	/**
	 * Sort the roi features by their distance from the center of the region of interest
	 *
	 * @param roi			- region of interest
	 * @param roiFeatures	- valid roiFeatures output
	 * @param keypoints		- local feature keypoints
	 */
    static void sortByDistanceFromCenter(Rect2d& roi, vector<int32_t>* roiFeatures, vector<KeyPoint>* keypoints);

	/**
	 * Calculate the histogram
	 *
	 * @param img_		- the image
	 * @return The histogram
	 */
    static Mat calculateHistogram(Mat& img_);

	/**
	 * Utility function for creating single object trackers given the name.
	 *
	 * @param name - the name of the tracker to create
	 * @return pointer to the tracker object
	 */
    static cv::Ptr<cv::Tracker> createTrackerByName(cv::String name);

	/**
	 * Trim the rectangle to the margin denoted by the padding
	 *
	 * @param r			-
	 * @param rows		-
	 * @param cols		-
	 * @param padding	-
	 */
    static bool trimRect(Rect2d& r, int32_t rows, int32_t cols, int32_t padding);

    /**
     *
     */
    static bool stabiliseRect(Mat& templateMatch, Rect2d& proposed);

	/**
	 * Use localised template matching to stabilise the detection.
	 */
    static bool stabiliseRect(Mat& frame, Rect2d templ_r, Rect2d& proposed);

	/**
	 *
	 */
    static bool _stabiliseRect(Mat& frame, Rect2d templ_r, Rect2d& proposed);

	/**
	 *
	 */
	static Rect2d shiftRect(Rect2d box, Point2f first, Point2f second);

	/**
	 *
	 */
	static void getListKeypoints(vector<KeyPoint>* keypoints, IntArrayList* list, vector<KeyPoint>* out);

	/**
	 *
	 */
	static void getVectorKeypoints(vector<KeyPoint>* keypoints, vector<int32_t>* list, vector<KeyPoint>* out);

	/**
	 *
	 */
	static Mat drawKeyPoints(Mat& in, vector<KeyPoint>* points, Scalar colour, int32_t type);

	/**
	 *
	 */
	static Mat getDescriptorDataset(Mat& descriptors, vector<KeyPoint>* keypoints, bool includeAngle, bool includeOctave);

	/**
	 *
	 */
	static void display(char const* screen, const InputArray& m);

	/**
	 *
	 */
	static Mat getColourDataset(Mat& f, vector<KeyPoint>* pts);

	/**
	 *
	 */
	static void quickSortByDistance(vector<int32_t>* roiFeatures, vector<double>* distances, int low, int high);

	/**
	 *
	 */
	static double calcDistanceL1(Point2f f1, Point2f f2);

	/**
	 *
	 */
	static void getSelectedKeypointsDescriptors(Mat& desc, IntArrayList* indices, Mat& out);

	/**
	 * Given a Rectangle and a reference feature size, use a target feature size to
	 * scale the rectagle.
	 */
	static Rect2d scaleRectangle(Rect2d in_r, double size_1, double size_2);
};
}
#endif // VOCUTILS_H
