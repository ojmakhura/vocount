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
UMat& LocatedObject::getBoxImage()
{
    return boxImage;
}

void LocatedObject::setBoxImage(UMat val)
{
    //boxImage = val;
    val.copyTo(boxImage);
    cvtColor(boxImage, boxGray, COLOR_RGB2GRAY);
}

///
/// boxGray
///
UMat& LocatedObject::getBoxGray()
{
    return boxGray;
}

void LocatedObject::setBoxGray(UMat boxGray)
{
	//this->boxGray = boxGray;
	boxGray.copyTo(this->boxGray);
}

///
/// histogram
///
Mat& LocatedObject::getHistogram()
{
    return histogram;
}

void LocatedObject::setHistogram(Mat val)
{
	val.copyTo(histogram);
    //histogram = val;
}

///
/// points
///
set<int32_t>* LocatedObject::getPoints()
{
    return &points;
}

void LocatedObject::setPoints(set<int32_t> val)
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
 * Check the rectangle already exists
 */
int32_t LocatedObject::rectExist(vector<LocatedObject>& structures, LocatedObject& bst)
{

    double maxIntersect = 0.0;
    int32_t maxIndex = -1;

    for(uint i = 0; i < structures.size(); i++)
    {
        Rect r2 = bst.getBox() & structures[i].getBox();
        double sect = ((double)r2.area()/bst.box.area()) * 100;
        if(sect > maxIntersect)
        {
            maxIndex = i;
            maxIntersect = sect;
        }
    }

    if(maxIntersect > 50.0)
    {
        return maxIndex;
    }

    return -1;
}

bool LocatedObject::createNewBoxStructure(KeyPoint first_p, KeyPoint second_p, LocatedObject& mbs, LocatedObject& n_mbs, UMat& frame)
{
	Mat _frame = frame.getMat(ACCESS_RW);
	Rect2d n_rect = VOCUtils::shiftRect(mbs.getBox(), first_p.pt, second_p.pt);

	Rect2d bx = mbs.getBox();
	VOCUtils::stabiliseRect(_frame, bx, n_rect);
	mbs.setBox(bx);
	n_mbs.setBox(n_rect);
	VOCUtils::trimRect(n_rect, frame.rows, frame.cols, 0);

	if(n_rect.height < 1 || n_rect.width < 1){
		return false;
	}

	double ratio = n_mbs.box.area()/n_rect.area();
	if(ratio < 0.2){
		return false;
	}

	n_mbs.setBoxImage(frame(n_rect));
	Mat h = VOCUtils::calculateHistogram(n_mbs.getBoxImage());
	n_mbs.setHistogram(h);
	UMat gr;
	cvtColor(n_mbs.getBoxImage(), gr, COLOR_RGB2GRAY);
	n_mbs.setBoxGray(gr);
	n_mbs.setHistogramCompare(compareHist(mbs.getHistogram(), n_mbs.getHistogram(), CV_COMP_CORREL));
	n_mbs.setMomentsCompare(matchShapes(mbs.getBoxGray(), n_mbs.getBoxGray(), CONTOURS_MATCH_I3, 0));

	return true;
}

/************************************************************************************
 *   PRIVATE FUNCTIONS
 ************************************************************************************/
};
