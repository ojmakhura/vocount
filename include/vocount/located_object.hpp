#ifndef LOCATEDOBJECT_H_
#define LOCATEDOBJECT_H_

#include <opencv/cv.hpp>
#include <map>
#include <set>

using namespace cv;
using namespace std;

namespace vocount
{

class LocatedObject
{
public:
    LocatedObject();
    virtual ~LocatedObject();
    LocatedObject(const LocatedObject& other);

    /************************************************************************************
     *   GETTERS AND SETTERS
     ************************************************************************************/

	///
	/// box
	///
    Rect2d getBox();
    void setBox(Rect2d val);

	///
	/// boxImage
	///
    UMat& getBoxImage();
    void setBoxImage(UMat val);

	///
	/// boxGray
	///
    UMat& getBoxGray();
    void setBoxGray(UMat boxGray);

	///
	/// histogram
	///
    Mat& getHistogram();
    void setHistogram(Mat val);

	///
	/// points
	///
    set<int32_t>* getPoints();
    void setPoints(set<int32_t> val);

	///
	/// histogramCompare
	///
    double getHistogramCompare();
    void setHistogramCompare(double histCompare);

	///
	/// momentsCompare
	///
    double getMomentsCompare();
    void setMomentsCompare(double momentsCompare);

	///
	/// matchPoint
	///
    Point getMatchPoint();
    void setMatchPoint(Point matchPoint);

    /************************************************************************************
     *   PUBLIC FUNCTIONS
     ************************************************************************************/

     /**
      *
      */
     void addToPoints(int32_t p);

     /**
      *
      */
     void removeFromPoints(int32_t p);

     /**
      *
      */
     static int32_t rectExist(vector<LocatedObject>& structures, LocatedObject& bst);

     /**
      *
      */
     static bool createNewBoxStructure(KeyPoint first_p, KeyPoint second_p, LocatedObject& mbs, LocatedObject& n_mbs, UMat& frame);

private:
    Rect2d box;
    UMat boxImage;
    UMat boxGray;
    Mat histogram;
    set<int32_t> points;
    double histogramCompare;
    double momentsCompare;
    Point matchPoint;

    /************************************************************************************
     *   PRIVATE FUNCTIONS
     ************************************************************************************/									/// Template match location
};
//typedef LocatedObject LocatedObject;

typedef map<int32_t, vector<LocatedObject>> map_st;
};
#endif // LOCATEDOBJECT_H_
