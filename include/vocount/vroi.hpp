#ifndef VROI_H_
#define VROI_H_
#include <opencv2/core.hpp>

using namespace cv;
//using namespace std;
//using namespace vocount;

namespace vocount
{


/**
 * This class manages a region of interest. It allows for creation through a cv:Rect2d or
 * through 4 Point2f objects while also including the translation and rotation capabilities
 */
class VRoi
{
public:
    /** Default constructor */
    VRoi();
    VRoi(Rect2d box);
    VRoi(Point2f p1, Point2f p2, Point2f p3, Point2f p4);
    VRoi(const VRoi& vroi);

    /** Default destructor */
    virtual ~VRoi();

    /**********************************************************************************************************************
     *   GETTERS AND SETTERS FUNCTIONS
     **********************************************************************************************************************/
    
    /**
     * box
     */ 
    Rect2d getBox();
    void setBox(const Rect2d& box);

    /**
     * p1
     */
    Point2f getP1();
    void setP1(const Point2f& p);

    /**
     * p2
     */
    Point2f getP2();
    void setP2(const Point2f& p);

    /**
     * p3
     */
    Point2f getP3();
    void setP3(const Point2f& p);

    /**
     * p4
     */
    Point2f getP4();
    void setP4(const Point2f& p);

    /**********************************************************************************************************************
     *   PUBLIC FUNCTIONS
     **********************************************************************************************************************/
	
    /**
     * Rotate the rectangle around the center.
     * 
     * @param angle : the angle in degrees.
     */
    void rotate(double angle); 

    VRoi& operator=(const VRoi& roi);    

private:
    Rect2d box;                     /// The reactangle that contains the points
    Point2f p1, p2, p3, p4;         /// Coners of the ROI

    /**********************************************************************************************************************
     *   PRIVATE FUNCTIONS
     **********************************************************************************************************************/
    Point2f rotatePoint(Point2f center, Point2f p, double angle);
    void comparePoints(Point2f p, Point2f& br, Point2f& tl);
    void copy(const VRoi& roi);
    void getPointsFromRect(Rect2d& box);
};
};
#endif // VROI_H_
