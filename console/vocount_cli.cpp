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
			"     [-f]       						# filtered keypoints\n"
			"     [-c]       						# cluster analysis method \n"
			"     [-df]       						# Combine descriptor clustering and filtered descriptors clustering\n"
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
	
	if(parser.has("d") || parser.has("df")){
		printf("*** Raw descriptor clustering activated\n");
		settings.dClustering = true;
	}
	
	if(parser.has("f") || parser.has("df")){
		printf("*** Filtered descriptor clustering activated\n");
		settings.fdClustering = true;
	}
	
	if(parser.has("df")){
		printf("*** Will combine descriptors and filtered descriptors results\n");
		settings.dfClustering = true;
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


/**
 * Using the map of clusters, allow the user to select the minPts
 * they want.
 * 
 */
void consolePreviewColours(Mat& frame, vector<KeyPoint>& keypoints, map<int, IntIntListMap* >& clusterMaps, vector<int32_t>& validities, int32_t& autoChoice){
	
	COLOURS c;
	std::string choiceStr = "yes";
	cout << "minPts = " << autoChoice << " has been detected. Use it? (yes/no): ";
	cin >> choiceStr;
		
	if(choiceStr.compare("yes") == 0 || choiceStr.compare("Yes") == 0 || choiceStr.compare("YES") == 0){
		return;
	}
	
	bool done = false;
	while(!done){
		cout << "-------------------------------------------------------------------------------" << endl;
		cout << "List of results \nminPts\t\tNumber of Clusters\t\tValidity" << endl;
		for(map<int, IntDoubleListMap* >::iterator it = clusterMaps.begin(); it != clusterMaps.end(); ++it){
			cout << it->first << "\t\t" << g_hash_table_size(it->second) << "\t\t" << validities[it->first - 3] << endl;
		}
		
		int32_t sel;
		cout << "Select minPts to preview: ";
		cin >> sel;
		cout << "Use 'n' to step through clusters and 'q' to exit preview." << endl;
		map<int, IntDoubleListMap* >::iterator it = clusterMaps.find(sel);
		
		if(it == clusterMaps.end()){
			cout << "minPts = " << sel << " is not in the results." << endl;
			continue;
		}
		
		IntIntListMap* tmp_map = it->second;
		
		GHashTableIter iter;
		gpointer key;
		gpointer value;
		g_hash_table_iter_init (&iter, tmp_map);

		while (g_hash_table_iter_next (&iter, &key, &value)){
			IntArrayList* list = (IntArrayList*)value;
			int32_t* k = (int32_t *)key;
			vector<KeyPoint> kps;
			getListKeypoints(keypoints, list, kps);
			Mat m = drawKeyPoints(frame, kps, c.red, -1);
			// print the choice images
			String imName = "choice_cluster_";
			imName += std::to_string(*k).c_str();	
			
			if(*k != 0){
				String windowName = "Choose ";
				windowName += std::to_string(*k).c_str();
				windowName += "?";
				display(windowName.c_str(), m);				
				
				// Listen for a key pressed
				char c = ' ';
				while(true){
					if (c == 'n') { // next cluster
						
						break;
					} else if (c == 'q'){ // stop preview
						break;
					}
					c = (char) waitKey(20);
				}
				destroyWindow(windowName.c_str());
				if(c == 'q'){
					break;
				}
			}
		}
		
		cout << "You previewed minPts = " << sel << ". Do you want to use it? (yes/no): ";		
		cin >> choiceStr;
		
		if(choiceStr.compare("yes") == 0 || choiceStr.compare("Yes") == 0 || choiceStr.compare("YES") == 0){
			cout << "Chosen " << sel << endl;
			autoChoice = sel;
			done = true;
		}
	}
	cout << "-------------------------------------------------------------------------------" << endl;
}


int main(int argc, char** argv) {
	ocl::setUseOpenCL(true);
	Mat frame;
	VideoCapture cap;
    vocount vcount;
    vsettings settings;
    //selection_t colourSel;
    settings.isConsole = true;
    vcount.colourSel.minPts = -1;
	vcount.detector = SURF::create(MIN_HESSIAN);
	
	cv::CommandLineParser parser(argc, argv,
					"{help ||}{o||}{n|1|}"
					"{v||}{video||}{w|1|}{s||}"
					"{c||}{t||}{l||}{ta|BOOSTING|}"
					"{d||}{f||}{df||}"
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
		framed f;
		vcount.detector->detectAndCompute(frame, Mat(), f.keypoints, f.descriptors);
		/**
		 * Finding the colour model for the current frame
		 * 
		 */ 
		if(vcount.frameCount == 0 && (settings.isClustering || settings.fdClustering)){
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Detecting Colour Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			printf("Finding proper value of minPts\n");
			map<int, IntIntListMap* > clusterMaps;
			vector<int32_t> validities = trainColourModel(vcount.colourSel, frame, f.keypoints, clusterMaps, vcount.trainingFile, settings.isConsole);	
			consolePreviewColours(frame, f.keypoints, clusterMaps, validities, vcount.colourSel.minPts);
			getLearnedColourModel(vcount.colourSel, clusterMaps, validities);
			//cout << " Selected ............ " << vcount.colourSel.minPts << endl;
			chooseColourModel(frame, f.descriptors, f.keypoints, vcount.colourSel);
			//consolePreviewColours(frame, keypoints, clusterMaps, vcount.colourSel.minPts);
			if(vcount.trackingFile.is_open()){
				vcount.trackingFile << 1 << "," << f.keypoints.size() << "," << vcount.colourSel.selectedDesc.rows << "," << vcount.colourSel.minPts << "," << vcount.colourSel.numClusters << "," << vcount.colourSel.validity << endl;
			}
		}
		
		// Listen for a key pressed
		char c = (char) waitKey(20);
		if (c == 'q') {
			break;
		} else if (c == 's') { // select a roi if c has een pressed or if the program was run with -s option
			settings.selectROI = true;
		} 
		processFrame(vcount, settings, f, frame);	
		//break;
		//if(vcount.frameCount == 4){
		//	break;
		//}
	}
	
	finalise(vcount);
	return 0;
}


