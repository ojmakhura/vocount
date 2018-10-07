#include "vocount/located_object.hpp"
#include "vocount/vocutils.hpp"

namespace vocount
{

LocatedObject::LocatedObject()
{
}

LocatedObject::~LocatedObject()
{

}

LocatedObject::LocatedObject(const LocatedObject& other)
{
    this->box = other.box;
    this->boxImage = other.boxImage.clone();
    histogram = other.histogram.clone();
    points = other.points;
    histogramCompare = other.histogramCompare;
    momentsCompare = other.momentsCompare;
    matchPoint = other.matchPoint;										/// Template match location
}

/************************************************************************************
 *   GETTERS AND SETTERS
 ************************************************************************************/

///
/// box
///
Rect2d LocatedObject::getBox()
{
    return box;
}

void LocatedObject::setBox(Rect2d val)
{
    box = val;
}

///
/// boxImage
///
Mat& LocatedObject::getBoxImage()
{
    return boxImage;
}

void LocatedObject::setBoxImage(Mat& val)
{
    boxImage = val.clone();
    cvtColor(boxImage, boxGray, COLOR_RGB2GRAY);
}

///
/// boxGray
///
Mat& LocatedObject::getBoxGray()
{
    return boxGray;
}

void LocatedObject::setBoxGray(Mat& boxGray)
{
    this->boxGray = boxGray.clone();
}

///
/// histogram
///
Mat& LocatedObject::getHistogram()
{
    return histogram;
}

void LocatedObject::setHistogram(Mat& val)
{
    histogram = val.clone();
}

///
/// points
///
set<int32_t>* LocatedObject::getPoints()
{
    return &points;
}

void LocatedObject::setPoints(set<int32_t>& val)
{
    points = val;
}

///
/// histCompare
///
double LocatedObject::getHistogramCompare()
{
    return this->histogramCompare;
}

void LocatedObject::setHistogramCompare(double histCompare)
{
    this->histogramCompare = histCompare;
}

///
/// momentsCompare
///
double LocatedObject::getMomentsCompare()
{
    return this->momentsCompare;
}

void LocatedObject::setMomentsCompare(double momentsCompare)
{
    this->momentsCompare = momentsCompare;
}

///
/// matchPoint
///
Point LocatedObject::getMatchPoint()
{
    return this->matchPoint;
}

void LocatedObject::setMatchPoint(Point matchPoint)
{
    this->matchPoint = matchPoint;
}

/************************************************************************************
 *   PUBLIC FUNCTIONS
 ************************************************************************************/

/**
 * Add this point idx to points
 */
void LocatedObject::addToPoints(int32_t p)
{
    this->points.insert(p);
}


/**
 *
 */
void LocatedObject::removeFromPoints(int32_t p)
{
    this->points.erase(p);
}

/**
 *
 */
void LocatedObject::addLocatedObject(vector<LocatedObject>* locatedObjects, LocatedObject* newObject)
{

    int32_t idx = LocatedObject::rectExist(locatedObjects, newObject);

    if(idx == -1)  // the structure does not exist
    {
        locatedObjects->push_back(*newObject);
    }
    else
    {
        LocatedObject& strct = locatedObjects->at(idx);

        // Find out which structure is more similar to the original
        // by comparing the moments
        if(newObject->getMomentsCompare() < strct.getMomentsCompare())
        {
            strct.setBox(newObject->getBox());
            strct.setMatchPoint(newObject->getMatchPoint());
            strct.setBoxImage(newObject->getBoxImage());
            strct.setBoxGray(newObject->getBoxGray());
            strct.setHistogram(newObject->getHistogram());
            strct.setHistogramCompare(newObject->getHistogramCompare());
            strct.setMomentsCompare(newObject->getMomentsCompare());
        }

        strct.getPoints()->insert(newObject->getPoints()->begin(), newObject->getPoints()->end());
    }
}

/**
 * Check the rectangle already exists
 */
int32_t LocatedObject::rectExist(vector<LocatedObject>* locatedObjects, LocatedObject* newObject)
{

    double maxIntersect = 0.0;
    int32_t maxIndex = -1;

    for(uint i = 0; i < locatedObjects->size(); i++)
    {
        Rect2d intersection = newObject->getBox() & locatedObjects->at(i).getBox();
        //Rect2d _union = newObject->getBox() | locatedObjects->at(i).getBox();

        double sect = ((double)intersection.area() / locatedObjects->at(i).getBox().area());
        if(sect > maxIntersect)
        {
            maxIndex = i;
            maxIntersect = sect;
        }
    }

    if(maxIntersect > 0.5)
    {
        return maxIndex;
    }

    return -1;
}

bool LocatedObject::createNewLocatedObject(KeyPoint first_p, KeyPoint second_p, LocatedObject* mbs, LocatedObject* n_mbs, Mat& frame)
{
    Rect2d n_rect = VOCUtils::shiftRect(mbs->getBox(), first_p.pt, second_p.pt);
    //n_rect = VOCUtils::scaleRectangle(n_rect, first_p.size, second_p.size);

    Rect2d bx = mbs->getBox();
    VOCUtils::stabiliseRect(frame, bx, n_rect);
    mbs->setBox(bx);
    n_mbs->setBox(n_rect);
    VOCUtils::trimRect(n_rect, frame.rows, frame.cols, 0);

    if(n_rect.height < 1 || n_rect.width < 1)
    {
        return false;
    }

    double ratio = n_mbs->getBox().area()/n_rect.area();
    if(ratio < 0.2)
    {
        return false;
    }

    Mat t_frame = frame(n_rect);
    n_mbs->setBoxImage(t_frame);
    Mat h = VOCUtils::calculateHistogram(n_mbs->getBoxImage());
    n_mbs->setHistogram(h);
    Mat gr;
    cvtColor(n_mbs->getBoxImage(), gr, COLOR_RGB2GRAY);
    n_mbs->setBoxGray(gr);
    n_mbs->setHistogramCompare(compareHist(mbs->getHistogram(), n_mbs->getHistogram(), CV_COMP_CORREL));
    n_mbs->setMomentsCompare(matchShapes(mbs->getBoxGray(), n_mbs->getBoxGray(), CONTOURS_MATCH_I3, 0));

    return true;
}

/************************************************************************************
 *   PRIVATE FUNCTIONS
 ************************************************************************************/
};
