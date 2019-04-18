#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include "vocount/vocutils.hpp"
#include <omp.h>

namespace vocount
{

/**
 * Find the roi features and at the same time find the central feature.
 *
 * @param  -
 * @param  -
 */
set<int32_t> VOCUtils::findValidROIFeatures(vector<KeyPoint>& keypoints, Rect2d& roi, vector<int32_t>& roiFeatures, vector<int32_t>& labels)
{
    Rect2d& r = roi;
    Point2f p;
    set<int32_t> roiClusters;
    p.x = (r.x + r.width)/2.0f;
    p.y = (r.y + r.height)/2.0f;
    for(uint i = 0; i < keypoints.size(); ++i)
    {
        int32_t label = labels.at(i);
        if(roi.contains(keypoints.at(i).pt) && label != 0)
        {
            roiClusters.insert(label);
            roiFeatures.push_back(i);
        }
    }

    return roiClusters;
}

/**
 * Find the roi features and at the same time find the central feature.
 *
 * @param  -
 * @param  -
 */
void VOCUtils::findROIFeatures(vector<KeyPoint>& keypoints, Rect2d& roi, vector<int32_t>& roiFeatures)
{
    Rect2d& r = roi;
    Point2f p;

    p.x = (r.x + r.width)/2.0f;
    p.y = (r.y + r.height)/2.0f;

    for(uint i = 0; i < keypoints.size(); ++i)
    {
        if(roi.contains(keypoints.at(i).pt))
        {
            roiFeatures.push_back(i);
        }
    }
}


// A utility function to swap two elements
void swap(int* a, int* b)
{
    int t = *a;
    *a = *b;
    *b = t;
}

// A utility function to swap two elements
void swap(double* a, double* b)
{
    double t = *a;
    *a = *b;
    *b = t;
}

/* This function takes last element as pivot, places
   the pivot element at its correct position in sorted
    array, and places all smaller (smaller than pivot)
   to left of pivot and all greater elements to right
   of pivot */
int partition (vector<int32_t>& roiFeatures, vector<double>& distances, int low, int high)
{
    double pivot = distances.at(high);    // pivot
    int i = (low - 1);  // Index of smaller element

    for (int j = low; j <= high- 1; j++)
    {
        // If current element is smaller than or
        // equal to pivot
        if (distances.at(j) <= pivot)
        {
            i++;    // increment index of smaller element
            swap(&(distances.at(i)), &(distances.at(j)));
            swap(&(roiFeatures.at(i)), &(roiFeatures.at(j)));
        }
    }
    swap(&(distances.at(i + 1)), &(distances.at(high)));
    swap(&(roiFeatures.at(i + 1)), &(roiFeatures.at(high)));
    return (i + 1);
}

/**
 * The main function that implements QuickSort using the distance
 *
 * @param roiFeatures --> vector to be sorted
 * @param distances --> vector to sort by,
 * @param low  --> Starting index,
 * @param high  --> Ending index
 */
void VOCUtils::quickSortByDistance(vector<int32_t>& roiFeatures, vector<double>& distances, int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, distances[p] is now
           at right place */
        int pi = partition(roiFeatures, distances, low, high);

        // Separately sort elements before
        // partition and after partition
        quickSortByDistance(roiFeatures, distances, low, pi - 1);
        quickSortByDistance(roiFeatures, distances, pi + 1, high);
    }
}


/**
 *
 */
double VOCUtils::calcDistanceL1(Point2f f1, Point2f f2){
	double diff = f1.x - f2.x;
	double sum = diff * diff;

	diff = f1.y - f2.y;
	sum += diff * diff;

	return sum;
}

/**
 * Sort roi features by the distance from the center
 *
 * @param  -
 * @param  -
 */
void VOCUtils::sortByDistanceFromCenter(Rect2d& roi, vector<int32_t>& roiFeatures, vector<KeyPoint>& keypoints)
{
    vector<double> distances;
    Point2f p;

    p.x = (roi.x + roi.width)/2.0f;
    p.y = (roi.y + roi.height)/2.0f;

    for(size_t i = 0; i < roiFeatures.size(); i++)
    {
        KeyPoint& r_kp = keypoints.at(roiFeatures.at(i));
        double distance = calcDistanceL1(p, r_kp.pt);
        distances.push_back(distance);
        //cout << roiFeatures[i] << ", " << distance << endl;
    }

    quickSortByDistance(roiFeatures, distances, 0, roiFeatures.size()-1);
}

/**
 *
 *
 * @param  -
 * @param  -
 */
Mat VOCUtils::calculateHistogram(Mat& img_)
{
    Mat hsv, hist;
    cvtColor(img_, hsv, COLOR_BGR2HSV );

    /// Using 50 bins for hue and 60 for saturation
    int h_bins = 50;
    int s_bins = 60;
    int histSize[] = { h_bins, s_bins };

    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };

    const float* ranges[] = { h_ranges, s_ranges };

    // Use the o-th and 1-st channels
    int channels[] = { 0, 1 };
    /// Calculate the histograms for the HSV images
    calcHist( &hsv, 1, channels, Mat(), hist, 2, histSize, ranges, true, false );
    normalize( hist, hist, 0, 1, NORM_MINMAX, -1, Mat() );
    //bst.hist = bst.hist.clone();

    return hist;
}

/**
 *
 *
 * @param  -
 * @param  -
 */
cv::Ptr<cv::Tracker> VOCUtils::createTrackerByName(cv::String name)
{
    cv::Ptr<cv::Tracker> tracker;

    if (name == "KCF")
    {
        tracker = cv::TrackerKCF::create();
    } else if (name == "TLD"){
        tracker = cv::TrackerTLD::create();
    } else if (name == "BOOSTING"){
        tracker = cv::TrackerBoosting::create();
    }else if (name == "MEDIAN_FLOW"){
        tracker = cv::TrackerMedianFlow::create();
    }else if (name == "MIL"){
        tracker = cv::TrackerMIL::create();
    }else if (name == "GOTURN"){
        tracker = cv::TrackerGOTURN::create();
    }else if (name == "MOSSE"){
        tracker = cv::TrackerMOSSE::create();
    }else if (name == "CSRT"){
        tracker = cv::TrackerCSRT::create();
    }else
        CV_Error(cv::Error::StsBadArg, "Invalid tracking algorithm name\n");

    cout << "Created " << name.c_str() << " tracker" << endl;
    return tracker;
}


/**
 * Trim the rectangle if it is within <padding> pixels of the frame edges
 *
 * @param r         - rectangle
 * @param rows      - frame height
 * @param cols      - frame width
 * @param padding   - the number of pixels at which to trim the rectangle
 */
bool VOCUtils::trimRect(Rect2d& r, int32_t rows, int32_t cols, int32_t padding)
{
    bool trimmed = false;
    if(r.x < padding)
    {
        r.width += r.x - padding;
        r.x = padding;
        trimmed = true;
    }

    if(r.y < padding)
    {
        r.height += r.y - padding;
        r.y = padding;
        trimmed = true;
    }

    if((r.x + r.width) >= cols - padding)
    {
        r.width = cols - r.x - padding;
        trimmed = true;
    }

    if((r.y + r.height) >= rows - padding)
    {
        r.height = rows - r.y - padding;
        trimmed = true;
    }

    return trimmed;
}

/**
 * Find the proper location of the best match for the rectangle
 * 
 */ 
bool VOCUtils::stabiliseRect(Mat& templateMatch, Rect2d& proposed)
{
    Rect2d new_r = proposed;
    int half_h = new_r.height/2;
    int half_w = new_r.width/2;
    new_r.x -= half_w/2;
    new_r.y -= half_h/2;
    new_r.width += half_w; //new_r.width;
    new_r.height += half_h; //new_r.height;

    trimRect(new_r, templateMatch.rows, templateMatch.cols, 0);
    if(new_r.height < 1 || new_r.width < 1)
    {
        return false;
    }

    Mat result = templateMatch(new_r);

    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    Point matchLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    matchLoc = minLoc;

    proposed.x = matchLoc.x + new_r.x;
    proposed.y = matchLoc.y + new_r.y;

    return true;
}

/**
 *
 *
 * @param  -
 * @param  -
 */
bool VOCUtils::stabiliseRect(Mat& frame, VRoi& templ_r, VRoi& proposed)
{
    Mat result;

    Rect2d new_r = proposed.getBox();
    int half_h = new_r.height/2;
    int half_w = new_r.width/2;
    new_r.x -= half_w/2;
    new_r.y -= half_h/2;
    new_r.width += half_w; //new_r.width;
    new_r.height += half_h; //new_r.height;

    trimRect(new_r, frame.rows, frame.cols, 0);
    if(new_r.height < 1 || new_r.width < 1)
    {
        return false;
    }

    Mat img = frame(new_r);
    Rect2d r_tmp = templ_r.getBox();
    //cout << "Before trimming " << r_tmp << endl;
    trimRect(r_tmp, frame.rows, frame.cols, 0);
    //cout << "After trimming " << r_tmp << endl;

    //if(r_tmp.height < 1 || r_tmp.width < 1)
    //{
        //return false;
    //}

    Mat templ = frame(r_tmp);

    if(img.rows < templ.rows && img.cols < templ.cols)
    {
        return false;
    }

    int result_cols =  img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;

    if(result_rows < 2 || result_cols < 2)
    {
        return false;
    }

    result.create( result_rows, result_cols, CV_32FC1 );
    matchTemplate( img, templ, result, TM_SQDIFF);
    normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    Point matchLoc;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    matchLoc = minLoc;

    proposed.getBox().x = matchLoc.x + new_r.x;
    proposed.getBox().y = matchLoc.y + new_r.y;

    return true;
}


/**
 * Given the frame and the keypoints, a Mat object is created based on
 * the colours at the keypoint locations.
 *
 * @param  -
 * @param  -
 */
bool VOCUtils::_stabiliseRect(Mat& frame, VRoi& templ_r, VRoi& proposed)
{
    Mat result;
    Rect2d r_tmp = templ_r.getBox();
    trimRect(r_tmp, frame.rows, frame.cols, 0);
    if(r_tmp.height < 1 || r_tmp.width < 1)
    {
        return false;
    }

    Mat templ = frame(r_tmp);
    int result_cols =  frame.cols - templ.cols + 1;
    int result_rows = frame.rows - templ.rows + 1;

    if(result_rows < 2 || result_cols < 2)
    {

        return false;
    }

    Rect2d& rec = proposed.getBox();
    result.create( result_rows, result_cols, CV_32FC1 );
    matchTemplate( frame, templ, result, TM_SQDIFF);
    trimRect(rec, result.rows, result.cols, 0);

    if(rec.height < 1 || rec.width < 1)
    {
        return false;
    }

    Mat p_img = result(rec);

    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    minMaxLoc( p_img, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

    int half_h = p_img.rows/2;
    int half_w = p_img.cols/2;
    Point half_p(half_w, half_h);
    Point diff_p = minLoc - half_p;
    //cout << proposed.getBox() << " : " << minLoc << " - " << half_p  << " = " << diff_p << endl;

    proposed.getBox().x = proposed.getBox().x + diff_p.x;
    proposed.getBox().y = proposed.getBox().y + diff_p.y;
    //cout << proposed.getBox() << endl;

    return true;
}

/**
 * Given the frame and the keypoints, a Mat object is created based on
 * the colours at the keypoint locations.
 *
 * @param  -
 * @param  -
 */
VRoi VOCUtils::shiftRect(VRoi& vroi, Point2f first, Point2f second)
{
    Rect2d n_rect = vroi.getBox();
    Point2d pshift;
    pshift.x = second.x - first.x;
    pshift.y = second.y - first.y;
    n_rect = n_rect + pshift;

    /// Update the points
    VRoi n_roi(n_rect);
    Point2f p_tmp = vroi.getP1();
    p_tmp.x = p_tmp.x +  pshift.x;
    p_tmp.y = p_tmp.y +  pshift.y;
    n_roi.setP1(p_tmp);

    p_tmp = vroi.getP2();
    p_tmp.x = p_tmp.x +  pshift.x;
    p_tmp.y = p_tmp.y +  pshift.y;
    n_roi.setP2(p_tmp);

    p_tmp = vroi.getP3();
    p_tmp.x = p_tmp.x +  pshift.x;
    p_tmp.y = p_tmp.y +  pshift.y;
    n_roi.setP3(p_tmp);

    p_tmp = vroi.getP4();
    p_tmp.x = p_tmp.x +  pshift.x;
    p_tmp.y = p_tmp.y +  pshift.y;
    n_roi.setP4(p_tmp);

    return n_roi;
}


/**
 * Given the frame and the keypoints, a Mat object is created based on
 * the colours at the keypoint locations.
 *
 * @param  -
 * @param  -
 */
void VOCUtils::getListKeypoints(vector<KeyPoint>& keypoints, IntArrayList* list, vector<KeyPoint>& out)
{
    int32_t* dt = (int32_t *)list->data;
    for(int i = 0; i < list->size; i++)
    {
        int32_t idx = dt[i];
        out.push_back(keypoints.at(idx));
    }
}

/**
 * Given the frame and the keypoints, a Mat object is created based on
 * the colours at the keypoint locations.
 *
 * @param  -
 * @param  -
 */
void VOCUtils::getVectorKeypoints(vector<KeyPoint>& keypoints, vector<int32_t>& list, vector<KeyPoint>& out)
{
    for(size_t i = 0; i < list.size(); i++)
    {
        int32_t idx = list.at(i);
        out.push_back(keypoints.at(idx));
    }
}


Mat VOCUtils::drawKeyPoints(Mat& in, vector<KeyPoint>& points, Scalar colour, int32_t type)
{
    Mat x = in.clone();
    if(type == -1)
    {
        for(vector<KeyPoint>::iterator it = points.begin(); it != points.end(); ++it)
        {
            circle(x, Point(it->pt.x, it->pt.y), 4, colour, cv::FILLED, 8, 0);
        }
    }
    else
    {
        DrawMatchesFlags flag = DrawMatchesFlags::DEFAULT;

        if(type == 1)
        {
            flag = DrawMatchesFlags::DRAW_OVER_OUTIMG;
        } else if(type == 2)
        {
            flag = DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS;
        } else if(type == 3)
        {
            flag = DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
        }

        cv::drawKeypoints( in, points, x, Scalar::all(-1), flag);
    }

    return x;
}

/**
 *
 */
void VOCUtils::getSelectedKeypointsDescriptors(Mat& desc, IntArrayList* indices, Mat& out)
{
	int32_t *dt = (int32_t *)indices->data;
	for(int i = 0; i < indices->size; i++){
		out.push_back(desc.row(dt[i]));
	}
}

/**
 * Compose the descriptor dataset. The dataset may be augmented depending on
 * the additions provided.
 *
 * @param descriptors   -
 * @param keypoints     -
 * @param additions     - 
 */
Mat VOCUtils::getDescriptorDataset(Mat& descriptors, vector<KeyPoint>& keypoints, VAdditions additions)
{
    Mat dataset = descriptors.clone();

    if(additions == VAdditions::ANGLE || additions == VAdditions::BOTH)
    {
        Mat angles(descriptors.rows, 1, CV_32FC1);
        float* data = angles.ptr<float>(0);

        #pragma omp parallel for
        for(size_t i = 0; i < keypoints.size(); i++)
        {
            KeyPoint kp = keypoints.at(i);
            data[i] = (M_PI / 180) * kp.angle;
        }

        normalize(angles, angles, 0, 1, cv::NORM_MINMAX, CV_32FC1);
        hconcat(dataset, angles, dataset);
    }

    if(additions == VAdditions::SIZE || additions == VAdditions::BOTH)
    {
        Mat sizes(descriptors.rows, 1, CV_32FC1);
        float* data = sizes.ptr<float>(0);
        #pragma omp parallel for
        for(size_t i = 0; i < keypoints.size(); i++)
        {
            KeyPoint kp = keypoints.at(i);
            data[i] = kp.size;
        }

        normalize(sizes, sizes, 0, 1, cv::NORM_MINMAX, CV_32FC1);
        hconcat(dataset, sizes, dataset);
    }

    if(!dataset.isContinuous())
    {
        dataset = dataset.clone();
    }

    return dataset;
}


/**
 * Given the frame and the keypoints, a Mat object is created based on
 * the colours at the keypoint locations.
 *
 * @param f - frame
 * @param pts - local features keypoints
 */
Mat VOCUtils::getColourDataset(Mat& f, vector<KeyPoint>& pts)
{
    Mat m(pts.size(), 3, CV_32FC1);
    Mat tmpf;
    GaussianBlur(f, tmpf, Size(3, 3), 0, 0 );
    //medianBlur(f, tmpf, 3);
    //tmpf = f;
    float* data = m.ptr<float>(0);
    #pragma omp parallel for
    for(size_t i = 0; i < pts.size(); i++)
    {
        Point2f pt = pts.at(i).pt;

        Vec3b p = tmpf.at<Vec3b>(pt);
        int idx = i * 3;

        data[idx] = p.val[0];
        data[idx + 1] = p.val[1];
        data[idx + 2] = p.val[2];
    }
    m = m.clone();
    return m;
}

/**
 *
 * @param
 * @param
 */
void VOCUtils::display(char const* screen, const InputArray& m)
{
    if (!m.empty())
    {
        namedWindow(screen, WINDOW_AUTOSIZE);
        imshow(screen, m);
    }
}


/**
 * Given a Rectangle and a reference feature size, use a target feature size to
 * scale the rectagle.
 *
 * @param in_r input rectangle
 * @param size_1 original feature size
 * @param size_2 Size of the target
 */
Rect2d VOCUtils::scaleRectangle(Rect2d in_r, double size_1, double size_2)
{
    Rect2d r_out;

    double scale = size_1/size_2;
    r_out.height = in_r.height * scale;
    r_out.width = in_r.width * scale;
    double h_diff = (in_r.height - r_out.height) / 2;
    double w_diff = (in_r.width - r_out.width) / 2;

    r_out.x = in_r.x + w_diff;
    r_out.y = in_r.y + h_diff;

    return r_out;
}
};
