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
		settings.dClustering = true;
	}
	
	if(parser.has("i") || parser.has("di") || parser.has("dfi")){
		settings.isClustering = true;
	}
	
	if(parser.has("f") || parser.has("df") || parser.has("dfi")){
		settings.fdClustering = true;
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
    colourSel.minPts = -1;
	vcount.detector = SURF::create(1000);
	
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
			createOutputDirectories(vcount, settings);
		}
	}

    if( !cap.isOpened() ){
        printf("Could not open stream\n");
    	return -1;
    }

    while(cap.read(frame))
    {
		display("frame", frame);
		
		// Listen for a key pressed
		char c = (char) waitKey(20);
		if (c == 'q') {
			break;
		} else if (c == 's') { // select a roi if c has een pressed or if the program was run with -s option
			settings.selectROI = true;
			/*Mat f2 = frame.clone();
			Rect2d boundingBox = selectROI("Select ROI", f2);
			destroyWindow("Select ROI");
			f.rois.push_back(boundingBox);
				
			for(size_t i = 0; i < f.rois.size(); i++){
				vcount.trackers.push_back(createTrackerByName(settings.trackerAlgorithm));
				vcount.trackers[i]->init( frame, f.rois[i] );
			}
				
			//trackers.add(_trackers, f2, f.rois);
			vcount.roiExtracted = true;
			*/
		} 
		//detector->detectAndCompute(frame, Mat(), keypoints, descriptors);
		processFrame(vcount, settings, colourSel, frame);		
	}
	
	finalise(vcount);
	return 0;
}


