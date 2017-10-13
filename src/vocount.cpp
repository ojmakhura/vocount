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
#include <opencv2/tracking.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ximgproc/segmentation.hpp>
#include <ctime>
#include <string>
#include "vocount/process_frame.hpp"
#include "vocount/print_utils.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::ximgproc::segmentation;

static void help(){
	printf( "This is a programming for estimating the number of objects in the video.\n"
	        "Usage: vocount\n"
	        "     [-v][-video]=<video>         	   	# Video file to read\n"
	        "     [-o=<output dir>]     		   	# the directly where to write to frame images\n"
			"     [-n=<sample size>]       			# the number of frames to use for sample size\n"
			"     [-w=<dataset width>]       		# the number of frames to use for dataset size\n"
			"     [-t=<truth count dir>]			# The folder that contains binary images for each frame in the video with objects tagged \n"
			"     [-s]       						# select roi from the first \n"
			"     [-i]       						# interactive results\n"
			"     [-c]       						# cluster analysis method \n"
	        "\n" );
}




int main(int argc, char** argv) {
	ocl::setUseOpenCL(true);
	Mat frame;
	VideoCapture cap;
    //BoxExtractor box;
    vocount vcount;
    selection_t colourSel;
    colourSel.minPts = -1;
	String colourDir;
	String indexDir;
	String keypointsDir;
	String selectedDir;

    Ptr<Feature2D> detector = SURF::create(500);
	Ptr<Tracker> tracker;

	cv::CommandLineParser parser(argc, argv,
					"{help ||}{o||}{n|1|}"
					"{v||}{video||}{w|1|}{s||}"
					"{i||}{c||}{t||}{l||}");


	if (parser.has("help")) {
		help();
		return 0;
	}

	if(parser.has("t")){
		vcount.truth = getFrameTruth(parser.get<String>("t"));
	}

	if(!processOptions(vcount, parser, cap)){
		help();
		return -1;
	}

    if( !cap.isOpened() ){
        printf("Could not open stream\n");
    	return -1;
    }

	//Create the output directories
    if(parser.has("o")){
		createDirectory(vcount.destFolder, "");

		colourDir = createDirectory(vcount.destFolder, "colour");
		indexDir = createDirectory(vcount.destFolder, "index");
		keypointsDir = createDirectory(vcount.destFolder, "keypoints");
		selectedDir = createDirectory(vcount.destFolder, "selected");
    }

    while(cap.read(frame))
    {
		vcount.frameCount++;
    	String colourFrameDir;
    	String indexFrameDir;
    	String keypointsFrameDir;
    	String selectedFrameDir;

        if(parser.has("o")){
    		colourFrameDir = createDirectory(colourDir, to_string(vcount.frameCount));
    		indexFrameDir = createDirectory(indexDir, to_string(vcount.frameCount));
    		keypointsFrameDir = createDirectory(keypointsDir, to_string(vcount.frameCount));
    		selectedFrameDir = createDirectory(selectedDir, to_string(vcount.frameCount));
        }

		framed f, index_f, sel_f;
		bool clusterInspect = false;

		f.i = vcount.frameCount;
		index_f.i = f.i;
		sel_f.i = f.i;

		f.frame = frame.clone();
		index_f.frame = f.frame;
		sel_f.frame = f.frame;

		cvtColor(f.frame, f.gray, COLOR_BGR2GRAY);
		detector->detectAndCompute(frame, Mat(), f.keypoints, f.descriptors);

		if(colourSel.minPts == -1 && (parser.has("c") || parser.has("i"))){
			printf("Finding proper value of minPts\n");
			colourSel = detectColourSelectionMinPts(frame, f.descriptors, f.keypoints);
			printf("Finding value of minPts = %d with colourSel.selectedPts as %lu from %lu\n", colourSel.minPts, colourSel.selectedDesc.rows, f.keypoints.size());
		}

		// Listen for a key pressed
		char c = (char) waitKey(20);
		if (c == 'q') {
			break;
		} else if (c == 's' || (parser.has("s") && !vcount.roiExtracted)) { // select a roi if c has een pressed or if the program was run with -s option
			tracker = TrackerBoosting::create();

			if (tracker == NULL) {
				cout << "***Error in the instantiation of the tracker...***\n";
				return -1;
			}

			Mat f2 = frame.clone();
			f.roi = selectROI("Select ROI", f2);
			destroyWindow("Select ROI");

			//initializes the tracker
			if (!tracker->init(frame, f.roi)) {
				cout << "***Could not initialize tracker...***\n";
				return -1;
			}

			vcount.roiExtracted = true;

		} else if(c == 'i'){ // inspect clusters
			clusterInspect = true;
		}

		if (vcount.roiExtracted ){
			tracker->update(f.frame, f.roi);
			RNG rng(12345);
			Scalar value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255),	rng.uniform(0, 255));
			rectangle(frame, f.roi, value, 2, 8, 0);
		}

		display("frame", frame);

		if (!f.descriptors.empty()) {

			cout << "################################################################################" << endl;
			cout << "                              " << f.i << endl;
			cout << "################################################################################" << endl;
			// Create clustering dataset

			f.hasRoi = vcount.roiExtracted;
			findROIFeature(f, colourSel);
			Mat dset = getDescriptorDataset(vcount.frameHistory, vcount.step, f.descriptors);

			dset = f.descriptors.clone();

			String filename = "dataset.h";

			ofstream myfile;
			myfile.open(filename.c_str());

			myfile << "#ifndef DATASET_H_ \n#define DATASET_H_" << endl;
			myfile << "int rows = " << dset.rows << ";" << endl;
			myfile << "int cols = " << dset.cols << ";" << endl;
			myfile << "double dataset ["  << dset.rows * dset.rows << "] = {" << endl;
			printf("Dataset has %d rows and %d cols\n", dset.rows, dset.cols);
			for(int i = 0; i < dset.rows; i++){
				for(int j = 0; j < dset.cols; j++){
					//printf("Printing for (%d, %d) = %f\n", i, j, dset.at<float>(i, j));
					if(i == dset.rows-1 && j == dset.cols-1){
						myfile << dset.at<float>(i, j);
					} else{
						myfile << dset.at<float>(i, j) << ",";

					}
				}
				myfile << endl;
			}

			myfile << "};\n#endif" << endl;
			myfile.close();

			results_t res1 = do_cluster(dset, f.keypoints, vcount.step, 3, TRUE);
			res1.roiClusterPoints = mapSampleFeatureClusters(f.roiFeatures, res1.labels);
			generateFinalPointClusters(res1.finalPointClusters, res1.roiClusterPoints, res1.clusterKeyPoints);
			
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			res1.total = countPrint(res1.roiClusterPoints, res1.clusterKeyPoints, res1.cest, res1.selectedFeatures, res1.lsize);
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
			cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;

			generateClusterImages(f.frame, res1.finalPointClusters, res1.keyPointImages, res1.cest, res1.total, res1.lsize, res1.selectedFeatures);
			boxStructure(res1.finalPointClusters, f.keypoints, f.roi, res1.boxStructures, res1.keyPointImages);

			f.results[""] = res1;
			printData(vcount, f.frame, 	f.keypoints, f.roiFeatures, res1, f.i);
			if(parser.has("o")){
				printImages(keypointsFrameDir, res1.keyPointImages, vcount.frameCount);
			}

			/*if(vcount.frameHistory.size() > 0){

				// Do index based clustering
				if (parser.has("i")) {
					if(colourSel.minPts != -1){
						framed ff = vcount.frameHistory[vcount.frameHistory.size()-1];
						vector<KeyPoint> keyp(ff.keypoints.begin(), ff.keypoints.end());

						keyp.insert(keyp.end(), f.keypoints.begin(), f.keypoints.end());

						Mat dataset = getColourDataset(frame, keyp);

						hdbscan<float> scanis(_EUCLIDEAN, 2*colourSel.minPts);
						scanis.run(dataset.ptr<float>(), dataset.rows, dataset.cols, true);

						vector<int> f_labels(scanis.getClusterLabels().begin() + ff.keypoints.size(), scanis.getClusterLabels().end());
						vector<int> p_labels(scanis.getClusterLabels().begin(), scanis.getClusterLabels().begin() + ff.keypoints.size());
						set<int> cselclusters, sss(p_labels.begin(), p_labels.end());
						map_kp clusterKeyPoints;
						map_t clusterKeypointIdx;
						mapClusters(p_labels, clusterKeyPoints, clusterKeypointIdx, ff.keypoints);
						//vector<KeyPoint> ssP;
						for(ulong i = 0; i < p_labels.size(); i++){
							//int idx = colourSel.selectedPtsIdx[i];
							cselclusters.insert(p_labels[i]);
						}
						/****************************************************************************************************/
						/*printf("Selected cluster ");

						for(set<int>::iterator it = cselclusters.begin(); it != cselclusters.end(); it++){
							printf("%d ", *it);
						}
						printf("\n");

						for(map_kp::iterator it = clusterKeyPoints.begin(); it != clusterKeyPoints.end(); it++){
							Mat mm = drawKeyPoints(frame, it->second, Scalar(0, 0, 255), -1);
							String s1 = to_string(it->first);
							display(s1.c_str(), mm);
						}
						/****************************************************************************************************/

						//Mat mm1 = drawKeyPoints(frame, ssP, Scalar(0, 0, 255), -1);
						//display("ssP", mm1);
						/*printf(" sss.size() = %lu\n", sss.size());
						printf("f_labels has %lu and f.keypoints has %lu and cselclusters has %lu and colourSel.selectedPtsIdx has %lu\n", f_labels.size(), f.keypoints.size(), cselclusters.size(), colourSel.selectedDesc.rows);

						Mat ds;
						vector<KeyPoint> selP;
						vector<int> selPi, roiPts;
						for(size_t i = 0; i < f_labels.size(); i++){
							if(cselclusters.find(f_labels[i]) != cselclusters.end()){
								selP.push_back(f.keypoints[i]);
								selPi.push_back(i);

								if(f.roi.contains(f.keypoints[i].pt)){
									roiPts.push_back(roiPts.size());
								}

								if(ds.empty()){
									ds = f.descriptors.row(i);
								} else{
									ds.push_back(f.descriptors.row(i));
								}
							}
						}

						vector<KeyPoint> selPts;

						for (uint j = 0; j < colourSel.selectedClusters.size(); j++) {
							int cluster = colourSel.selectedClusters[j];
							vector<KeyPoint> pts = colourSel.clusterKeyPoints[cluster];
							selPts.insert(selPts.end(), pts.begin(), pts.end());
						}
						Mat mm = drawKeyPoints(frame, selPts, Scalar(0, 0, 255), -1);
						display("selP", mm);*/

						/*colourSel.selectedDesc = ds.clone();
						//colourSel.selectedPts = selP;
						colourSel.roiFeatures = roiPts;
						//colourSel.selectedPtsIdx = selPi;
						colourSel.clusterKeyPoints = clusterKeyPoints;
						colourSel.clusterKeypointIdx = clusterKeypointIdx;

						//printf("colourSel.selectedDesc has %d rows and colourSel.selectedPts has %lu size\n", colourSel.selectedDesc.rows, colourSel.selectedPts.size());

						vector<KeyPoint> selPts;

						for (uint j = 0; j < colourSel.selectedClusters.size();
								j++) {
							int cluster = colourSel.selectedClusters[j];
							vector<KeyPoint> pts = colourSel.clusterKeyPoints[cluster];
							selPts.insert(selPts.end(), pts.begin(), pts.end());
						}

						ds = getPointDataset(selPts);
						results_t rs = cluster(dataset, 3, true);
						rs.keypoints = selPts;
						set<int> ss(rs.labels.begin(), rs.labels.end());
						//printf("-------------- We found %lu objects by index points clustering.\n", ss.size() - 1);
						//mapClusters(rs.labels, rs.clusterKeyPoints, rs.clusterKeypointIdx, rs.keypoints);
						rs.roiClusterPoints = mapSampleFeatureClusters(colourSel.roiFeatures, rs.labels);
						generateClusterImages(f.frame, rs.clusterKeyPoints, rs.keyPointImages, rs.cest, rs.total, res1.lsize, res1.selectedFeatures);

						if(parser.has("o")){
							printf("--------------------- found %lu images\n", rs.keyPointImages.size());
							printImages(indexFrameDir, rs.keyPointImages, vcount.frameCount);
						}

						dataset = colourSel.selectedDesc.clone();
						results_t sel_r = cluster(dataset, 3, true);
						sel_r.keypoints = selPts;
						//hdbscan<float> sel_scan(_EUCLIDEAN, 3);
						//sel_scan.run(sel_r.dataset.ptr<float>(), sel_r.dataset.rows, sel_r.dataset.cols, true);
						//sel_r.labels = sel_scan.getClusterLabels();
						//mapClusters(sel_r.labels, sel_r.clusterKeyPoints, sel_r.clusterKeypointIdx, sel_r.keypoints);
						sel_r.roiClusterPoints = mapSampleFeatureClusters(colourSel.roiFeatures, sel_r.labels);
						generateClusterImages(f.frame, sel_r.clusterKeyPoints, sel_r.keyPointImages, sel_r.cest, sel_r.total, res1.lsize, res1.selectedFeatures);

						if(parser.has("o")){
							//printf("--------------------- found %lu images\n", sel_r.keyPointImages.size());
							printImages(selectedFrameDir, sel_r.keyPointImages, vcount.frameCount);
						}
						// update the colour selection so that for every new frame, colourSel is based on the previous frame.
						scanis.clean();
						//sel_scan.clean();
						//sc.clean();
					}


				}
			}*/
		}

		maintaintHistory(vcount, f);
	}

    if(vcount.print){
    	printStats(vcount.destFolder, vcount.stats);
    	printClusterEstimates(vcount.destFolder, vcount.clusterEstimates);
    }

	return 0;
}


