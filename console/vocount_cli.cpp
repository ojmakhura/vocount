//#include "hdbscan.hpp"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <opencv/cv.hpp>
#include <opencv/cxcore.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <ctime>
#include <string>
#include "vocount/process_frame.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ximgproc::segmentation;



int main(int argc, char** argv) {
	ocl::setUseOpenCL(true);
	Mat frame;
	VideoCapture cap;
    vocount vcount;
    vsettings settings;
    selection_t colourSel;
    colourSel.minPts = -1;
	vector<Rect2d> rois;
    Ptr<Feature2D> detector = SURF::create(1000);
    vector<Ptr<Tracker>> trackers;

	cv::CommandLineParser parser(argc, argv,
					"{help ||}{o||}{n|1|}"
					"{v||}{video||}{w|1|}{s||}"
					"{i||}{c||}{t||}{l||}{ta|BOOSTING|}"
					"{d||}{f||}{df||}{di||}{dfi||}");


	

	if(!processOptions(vcount, settings, parser, cap)){
		help();
		return -1;
	} else{
		if(settings.print){
			createOutputDirectories(vcount, settings)
		}
	}

    if( !cap.isOpened() ){
        printf("Could not open stream\n");
    	return -1;
    }

    while(cap.read(frame))
    {
		display("frame", frame);
		processKeypoints(vcount, settings, colourSel, frame);		
	}
	
	finalise(vcount);
	return 0;
}


