#include "vocount/vroi.hpp"
#include <cmath>

namespace vocount
{

    VRoi::VRoi()
    {

    }

    VRoi::VRoi(Rect2d box)
    {
        this->box = box;
        getPointsFromRect(box);
    }
    
    VRoi::VRoi(Point2f p1, Point2f p2, Point2f p3, Point2f p4)
    {
        //this->p1 = p1;
        //this->p2 = p2;
        //this->p3 = p3;
        //this->p4 = p4;

        //double x = p1.
    }

    VRoi::VRoi(const VRoi& vroi)
    {
        copy(vroi);
    }

    /** Default destructor */
    VRoi::~VRoi()
    {

    }

    /**********************************************************************************************************************
     *   GETTERS AND SETTERS FUNCTIONS
     **********************************************************************************************************************/
    
    /**
     * box
     */ 
    Rect2d VRoi::getBox()
    {
        return this->box;
    }

    void VRoi::setBox(const Rect2d& box)
    {
        this->box = box;
        getPointsFromRect(this->box);
    }

    /**
     * p1
     */
    Point2f VRoi::getP1()
    {
        return this->p1;
    }

    void VRoi::setP1(const Point2f& p)
    {
        this->p1 = p;
    }

    /**
     * p2
     */
    Point2f VRoi::getP2()
    {
        return p2;
    }

    void VRoi::setP2(const Point2f& p)
    {
        this->p2 = p;
    }

    /**
     * p3
     */
    Point2f VRoi::getP3()
    {
        return this->p3;
    }

    void VRoi::setP3(const Point2f& p)
    {
        this->p3 = p;
    }

    /**
     * p4
     */
    Point2f VRoi::getP4()
    {
        return p4;
    }

    void VRoi::setP4(const Point2f& p)
    {
        this->p4 = p;
    }

    /**********************************************************************************************************************
     *   PUBLIC FUNCTIONS
     **********************************************************************************************************************/
	
    /**
     * 
     * 
     */
    void VRoi::rotate(double angle)
    {
        double xc = box.x + box.width / 2;
        double yc = box.y + box.height / 2;

        Point2f center(xc, yc);
        p1 = rotatePoint(center, p1, angle);
        p2 = rotatePoint(center, p2, angle);
        p3 = rotatePoint(center, p3, angle);
        p4 = rotatePoint(center, p4, angle);

        Point2f tl, br;
        tl.x = br.x = p1.x;
        tl.y = br.y = p1.y;

        /// Check p2
        comparePoints(p2, br, tl);
        comparePoints(p3, br, tl);
        comparePoints(p4, br, tl);

        box = Rect2d(tl, br);
    }

    VRoi& VRoi::operator=(const VRoi& roi)
    {
        copy(roi);
        return *this;
    }

    /**********************************************************************************************************************
     *   PRIVATE FUNCTIONS
     **********************************************************************************************************************/
    
    Point2f VRoi::rotatePoint(Point2f center, Point2f p, double angle)
    {
        Point2f _p = p - center;
        Point2f p_;
        p_.x = _p.x * cos(angle) - _p.y * sin(angle);
        p_.y = _p.x * sin(angle) + _p.y * cos(angle);

        return p_ + center;
    }

    void VRoi::comparePoints(Point2f p, Point2f& br, Point2f& tl){
        if(p.x < tl.x)
        {
            tl.x = p.x;
        }

        if(p.x > br.x)
        {
            br.x = p.x;
        }

        if(p.y < tl.y)
        {
            tl.y = p.y;
        }

        if(p.y > br.y)
        {
            br.y = p.y;
        }
    }

    void VRoi::copy(const VRoi& roi)
    {
        this->box = roi.box;
        this->p1 = roi.p1;
        this->p2 = roi.p2;
        this->p3 = roi.p3;
        this->p4 = roi.p4;
    }

    void VRoi::getPointsFromRect(Rect2d& box)
    {
        // top left point
        Point2f tl = box.tl();
        
        //bottom right point
        Point2f br = box.br();

        this->p1 = tl;
        this->p2 = Point2f(br.x, tl.y);
        this->p3 = br;
        this->p4 = Point2f(tl.x, br.y);
    }

};