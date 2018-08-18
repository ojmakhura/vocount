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
#include <ctime>
#include <string>
#include "vocount/process_frame.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

static void help(){
	printf( "This is a programming for estimating the number of objects in the video.\n"
	        "Usage: vocount\n"
	        "     [-v][-video]=<video>         	   	# Video file to read\n"
	        "     [-o=<output dir>]     		   	# the directly where to write to frame images\n"
			"     [-n=<sample size>]       			# the number of frames to use for sample size\n"
			"     [-w=<dataset width>]       		# the number of frames to use for dataset size\n"
			"     [-t=<truth count dir>]			# The folder that contains binary images for each frame in the video with objects tagged \n"
			"     [-s]       						# select roi from the first \n"
			"     [-d]       						# raw descriptors\n"
			"     [-i]       						# image space clustering\n"
			"     [-f]       						# filtered keypoints\n"
			"     [-c]       						# cluster analysis method \n"
			"     [-df]       						# Combine descriptor clustering and filtered descriptors clustering\n"
			"     [-di]       						# Combine descriptor clustering and image index based clustering\n"
			"     [-dfi]       					# Combine descriptor clustering, filtered descriptors and index based clustering\n"
			"     [-rx]       					# roi x coordiate\n"
			"     [-ry]       					# roi y coordinate\n"
			"     [-rh]       					# roi height\n"
			"     [-rw]       					# roi width\n"
			"     [-e]       					# extend the box structures to include clusters not in the initial list\n"
			"     [-r]       					# rotate the rectangles\n"
			"     [-D]       					# Enable debug messages\n"
	        "\n" );
}

bool processOptions(vocount& vcount, vsettings& settings, CommandLineParser& parser, VideoCapture& cap){
		
	/*if (parser.has("help")) {
		help();
		return 0;
	}*/

	if (parser.has("o")) {
		settings.outputFolder = parser.get<String>("o");
		settings.print = true;
		printf("Will print to %s\n", settings.outputFolder.c_str());
	}

	if (parser.has("v") || parser.has("video")) {

		settings.inputVideo =
				parser.has("v") ?
						parser.get<String>("v") : parser.get<String>("video");
		cap.open(settings.inputVideo);
	} else {
		printf("You did not provide the video stream to open.");
		return false;
	}

	if (parser.has("w")) {

		String s = parser.get<String>("w");
		settings.step = atoi(s.c_str());
	} else {
		settings.step = 1;
	}

	if (parser.has("n")) {
		String s = parser.get<String>("n");
		settings.rsize = atoi(s.c_str());
	}

	if(parser.has("t")){
		settings.truthFolder = parser.get<String>("t");
		getFrameTruth(settings.truthFolder, vcount.truth);
	}
	
	if(parser.has("ta")){
		settings.trackerAlgorithm = parser.get<String>("ta");
	}
	
	if(parser.has("s")){
		settings.selectROI = true;
	}
	
	if(parser.has("d") || parser.has("df") || parser.has("di") || parser.has("dfi")){
		printf("*** Raw descriptor clustering activated\n");
		settings.dClustering = true;
	}
	
	if(parser.has("i") || parser.has("di") || parser.has("dfi")){
		printf("*** Image space clustering activated\n");
		settings.isClustering = true;
	}
	
	if(parser.has("f") || parser.has("df") || parser.has("dfi")){
		printf("*** Filtered descriptor clustering activated\n");
		settings.fdClustering = true;
	}
	
	if(parser.has("df")){
		printf("*** Will combine descriptors and filtered descriptors results\n");
		settings.dfClustering = true;
	}
	
	if(parser.has("di")){
		printf("*** Will combine descriptors and image space clustering results\n");
		settings.diClustering = true;
	}
	
	if(parser.has("dfi")){
		printf("*** Will combine descriptors, filtered descriptors and image space clustering results\n");
		settings.dfiClustering = true;
	}
	
	if(parser.has("e")){
		printf("*** Will extend the box structures\n");
		settings.extend = true;
	}
	
	if(parser.has("r")){
		printf("*** Will rotate the rectangles for rotational invariance\n");
		settings.rotationalInvariance = true;
	}
	
	if(parser.has("D")){
		printf("*** Debug enabled.\n");
		VO_DEBUG = true;
	}
	
	if(parser.has("rx") && parser.has("ry") && parser.has("rw") && parser.has("rh")){
		printf("*** ROI provided from command line\n");
		settings.selectROI = false;
		String s = parser.get<String>("rx");
		int x = atoi(s.c_str());
		s = parser.get<String>("ry");
		int y = atoi(s.c_str());
		s = parser.get<String>("rw");
		int w = atoi(s.c_str());
		s = parser.get<String>("rh");
		int h = atoi(s.c_str());
		
		vcount.roiExtracted = true;
		vcount.roi = Rect2d(x, y, w, h);;
	} else{
		vcount.roi = Rect2d(0, 0, 0, 0);		
	}
	
	return true;
}

int main(int argc, char** argv) {
	ocl::setUseOpenCL(true);
	Mat frame;
	VideoCapture cap;
    vocount vcount;
    vsettings settings;
    selection_t colourSel;
    vcount.isConsole = true;
    colourSel.minPts = -1;
	vcount.detector = SURF::create(MIN_HESSIAN);
	
	cv::CommandLineParser parser(argc, argv,
					"{help ||}{o||}{n|1|}"
					"{v||}{video||}{w|1|}{s||}"
					"{i||}{c||}{t||}{l||}{ta|BOOSTING|}"
					"{d||}{f||}{df||}{di||}{dfi||}"
					"{rx||}{ry||}{rw||}{rh||}{e||}{r||}{D||}");

	if(!processOptions(vcount, settings, parser, cap)){
		help();
		return -1;
	} else{
		if(settings.print){
			createOutputDirectories(vcount, settings);
		}
	}

    if( !cap.isOpened() ){
        printf("Could not open stream\n");
    	return -1;
    }

    while(cap.read(frame))
    {	
		// Listen for a key pressed
		char c = (char) waitKey(20);
		if (c == 'q') {
			break;
		} else if (c == 's') { // select a roi if c has een pressed or if the program was run with -s option
			settings.selectROI = true;
		} 
		processFrame(vcount, settings, colourSel, frame);	
		//break;
		//if(vcount.frameCount == 4){
		//	break;
		//}
	}
	
	finalise(vcount, colourSel);
	return 0;
}


