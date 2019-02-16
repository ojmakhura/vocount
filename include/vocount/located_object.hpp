#ifndef LOCATEDOBJECT_H_
#define LOCATEDOBJECT_H_

#include <opencv2/core.hpp>
#include <map>
#include <set>

#include "vocount/vroi.hpp"

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
    VRoi getBoundingBox();
    void setBoundingBox(VRoi val);

	///
	/// boxImage
	///
    Mat& getBoxImage();
    void setBoxImage(Mat& val);

	///
	/// boxGray
	///
    Mat& getBoxGray();
    void setBoxGray(Mat& boxGray);

	///
	/// histogram
	///
    Mat& getHistogram();
    void setHistogram(Mat& val);

	///
	/// points
	///
    set<int32_t>* getPoints();
    void setPoints(set<int32_t>& val);

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

	///
	/// matchTo
	///
    Rect2d getMatchTo();
    void setMatchTo(Rect2d matchTo);

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
     static int32_t rectExist(vector<LocatedObject>* locatedObjects, LocatedObject* newObject);

     /**
      *
      */
     static bool createNewLocatedObject(KeyPoint first_p, KeyPoint second_p, LocatedObject* existingObject, LocatedObject* newObject, Mat& frame);

     /**
      *
      */
     static void addLocatedObject(vector<LocatedObject>* locatedObjects, LocatedObject* newObject);

private:
    VRoi boundingBox;
    Mat boxImage;
    Mat boxGray;
    Mat histogram;
    set<int32_t> points;
    double histogramCompare;
    double momentsCompare;
    Point matchPoint;
    Rect2d matchTo;

    /************************************************************************************
     *   PRIVATE FUNCTIONS
     ************************************************************************************/									/// Template match location
};

typedef map<int32_t, vector<LocatedObject>> map_st;
};
#endif // LOCATEDOBJECT_H_
