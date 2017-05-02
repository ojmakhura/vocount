/*
 * box_extractor.h
 *
 *  Created on: 23 Oct 2016
 *      Author: junior
 */

#ifndef BOX_EXTRACTOR_HPP_
#define BOX_EXTRACTOR_HPP_
#include <opencv/cv.hpp>
using namespace cv;

namespace clustering {

class BoxExtractor {
public:
public:
  Rect2d extract(Mat img);
  Rect2d extract(const std::string& windowName, Mat img, bool showCrossair = true);

  struct handlerT{
    bool isDrawing;
    Rect2d box;
    Mat image;

    // initializer list
    handlerT(): isDrawing(false) {};
  }params;

private:
  static void mouseHandler(int event, int x, int y, int flags, void *param);
  void opencv_mouse_callback( int event, int x, int y, int , void *param );
};

} /* namespace hdbscan */

#endif /* BOX_EXTRACTOR_HPP_ */
