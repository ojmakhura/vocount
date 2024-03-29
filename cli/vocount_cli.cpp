//#include "hdbscan.hpp"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <ctime>
#include <string>
#include "vocount/vocounter.hpp"
#include "vocount/voprinter.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace vocount;

bool videoMode = true;
String imagePath;

static void help()
{
    printf( "This is a programming for estimating the number of objects in the video.\n"
            "Usage: vocount\n"
            "     [-v][-video]=<video>         	   	# Video file to read\n"
            "     [-image]=<image>           	   	# Image file to read\n"
            "     [-o=<output dir>]     		   	# the directly where to write to frame images\n"
            "     [-n=<sample size>]       			# the number of frames to use for sample size\n"
            "     [-w=<dataset width>]       		# the number of frames to use for dataset size\n"
            "     [-t=<truth count dir>]			# The folder that contains binary images for each frame in the video with objects tagged \n"
            "     [-s]       						# select roi from the first \n"
            "     [-d]       						# detect clusters in raw descriptors\n"
            "     [-c]       						# detect clusters in the colour model \n"
            "     [-f]       						# Filter descriptor clusters with the colour model \n"
            "     [-cm]       						# Combine descriptor clustering and filtered descriptors clustering\n"
            "     [-co]       						# print images that show all the clusters in the results without matching and bounding boxes\n"
            "     [-ct]       						# Track the colour model\n"
            "     [-rx]       					    # roi x coordiate\n"
            "     [-ry]       					    # roi y coordinate\n"
            "     [-rh]       					    # roi height\n"
            "     [-rw]       					    # roi width\n"
            "     [-r]       					    # add keypoint angles to dataset\n"
            "     [-R]       					    # rotate the rectangles\n"
            "     [-z]       					    # add sizes to dataset\n"
            "     [-D]       					    # Enable debug messages\n"
            "     [-O]       					    # Enable using minPts = 2 if no valid clustering results are detected\n"
            "     [-I=<number of iterations>] 	    # The number of iterations for extending cluster daisy chaining\n"
            "     [-minPts=<first minPts value>] 	# The first minPts value to try\n"
            "     [-eo                              # Print estimates only without images\n"
            "     [-fi                              # Print estimates and final images only.\n"
            "     [-pa                              # Print estimates and all the images.\n"
            "     [-tr=<Tracker name>               # The name of the SOT to use.\n"
            "     [-tp                              # Print tracker information\n"
            "\n" );
}

bool processOptions(vsettings& settings, CommandLineParser& parser)
{

    if (parser.has("o"))
    {
        settings.outputFolder = parser.get<String>("o");
        settings.print = true;
        printf("Will print to %s\n", settings.outputFolder.c_str());

        if(parser.has("eo"))
        {
            settings.outputType = OutputType::ESTIMATES;
            printf("*** Printing estimates only\n");
        } else if (parser.has("fi"))
        {
            settings.outputType = OutputType::FINALIMAGES;
            printf("*** Printing estimates and final images only\n");
        } else if (parser.has("pa"))
        {
            settings.outputType = OutputType::ALL;
            printf("*** Printing estimates, final and cluster images images only\n");
        }
    }

    if (parser.has("v") || parser.has("video") || parser.has("image"))
    {

        if(parser.has("v") || parser.has("video"))
        {
            settings.inputFile =
                parser.has("v") ?
                parser.get<String>("v") :  parser.get<String>("video");
        } else {
            settings.inputFile = parser.get<String>("image");
            videoMode = false;
        }
    }
    else
    {
        printf("You did not provide the video stream to open.");
        return false;
    }

    if (parser.has("w"))
    {
        String s = parser.get<String>("w");
        settings.step = atoi(s.c_str());
    }
    else
    {
        settings.step = 1;
    }

    if (parser.has("n"))
    {
        String s = parser.get<String>("n");
        settings.rsize = atoi(s.c_str());
    }

    if(parser.has("t"))
    {
        settings.truthFolder = parser.get<String>("t");
    }

    if(parser.has("tr"))
    {
        settings.trackerAlgorithm = parser.get<String>("tr");
    }

    if(parser.has("tp"))
    {
        settings.printTracking = true;
    }

    if(parser.has("s"))
    {
        settings.selectROI = true;
    }

    if(parser.has("d") || parser.has("cm") || parser.has("f"))
    {
        printf("*** Raw descriptor clustering activated\n");
        settings.descriptorClustering = true;
    }

    if(parser.has("c") || parser.has("cm") || parser.has("f") || parser.has("ct"))
    {
        printf("*** Colour model tracking activated\n");
        settings.colourModelTracking = true;
    }

    if(parser.has("c"))
    {
        printf("*** Colour model descriptor clustering activated\n");
        settings.colourModelClustering = true;
    }

    if(parser.has("f"))
    {
        printf("*** Filter descriptor clusters with colour model\n");
        settings.colourModelFiltering = true;
    }

    if(parser.has("cm"))
    {
        printf("*** Will combine descriptors and filtered descriptors results\n");
        settings.combine = true;

    }

    if(parser.has("co"))
    {
        printf("*** Will print clusters only\n");
        settings.clustersOnly = true;
    }

    if(parser.has("r") && parser.has("z"))
    {
        printf("*** Will add angles and sizes to descriptors.\n");
        settings.additions = VAdditions::BOTH;

    } else if(parser.has("r"))
    {
        printf("*** Will add angles to descriptors for rotational invariance\n");
        settings.additions = VAdditions::ANGLE;

    } else if(parser.has("z"))
    {
        printf("*** Will include sizes in the dataset\n");
        settings.additions = VAdditions::SIZE;
    } else
    {
        settings.additions = VAdditions::NONE;
    }

    if(parser.has("O"))
    {
        printf("*** Will use minPts = 2 if validity is less than 4\n");
        settings.overSegment = true;
    }

    if(parser.has("I"))
    {
        const String s = parser.get<String>("I");
        stringstream stmp(s.c_str());
        string word;
        while (getline(stmp, word, ',')) { 
            settings.iterations.push_back(atoi(word.c_str())); 
        } 
        
        cout << "*** Daisy-chaining iteration set to " << s << endl;
    }
    else
    {
        settings.iterations.push_back(0); 
    }

    if(parser.has("D"))
    {
        printf("*** Debug enabled.\n");
        VO_DEBUG = true;
    }

    if(parser.has("minPts"))
    {
        String s = parser.get<String>("minPts");
        stringstream stmp(s.c_str());
        string word;
        while (getline(stmp, word, ',')) { 
            settings.minPts.push_back(atoi(word.c_str())); 
        } 
        cout << "*** minPts = " << s << endl;
    } else {
        settings.minPts.push_back(3);
    }

    if(parser.has("rx") && parser.has("ry") && parser.has("rw") && parser.has("rh"))
    {
        printf("*** ROI provided from command line\n");
        settings.selectROI = false;
        String s = parser.get<String>("rx");
        settings.x = atoi(s.c_str());
        s = parser.get<String>("ry");
        settings.y = atoi(s.c_str());
        s = parser.get<String>("rw");
        settings.w = atoi(s.c_str());
        s = parser.get<String>("rh");
        settings.h = atoi(s.c_str());
    }

    return true;
}


/**
 * Using the map of clusters, allow the user to select the minPts
 * they want.
 *
 */
int32_t consolePreviewColours(Mat& frame, vector<KeyPoint>& keypoints, map<int32_t, IntIntListMap* >& clusterMaps, vector<int32_t>& validities, int32_t autoChoice)
{
    int32_t chosen = autoChoice;
    COLOURS c;
    std::string choiceStr = "yes";
    cout << "minPts = " << chosen << " has been detected. Use it? (yes/no): ";
    cin >> choiceStr;

    if(choiceStr.compare("yes") == 0 || choiceStr.compare("Yes") == 0 || choiceStr.compare("YES") == 0)
    {
        return chosen;
    }

    bool done = false;
    while(!done)
    {
        cout << "-------------------------------------------------------------------------------" << endl;
        cout << "List of results \nminPts\t\tNumber of Clusters\t\tValidity" << endl;
        for(map<int32_t, IntDoubleListMap* >::iterator it = clusterMaps.begin(); it != clusterMaps.end(); ++it)
        {
            cout << it->first << "\t\t" << g_hash_table_size(it->second)-1 << "\t\t" << validities.at(it->first - 3) << endl;
        }

        int32_t sel;
        cout << "Select minPts to preview: ";
        cin >> sel;
        cout << "Use 'n' to step through clusters and 'q' to exit preview." << endl;
        map<int32_t, IntDoubleListMap* >::iterator it = clusterMaps.find(sel);

        if(it == clusterMaps.end())
        {
            cout << "minPts = " << sel << " is not in the results." << endl;
            continue;
        }

        IntIntListMap* tmp_map = it->second;

        GHashTableIter iter;
        gpointer key;
        gpointer value;
        g_hash_table_iter_init (&iter, tmp_map);

        while (g_hash_table_iter_next (&iter, &key, &value))
        {
            ArrayList* list = (ArrayList*)value;
            int32_t* k = (int32_t *)key;
            vector<KeyPoint> kps;
            VOCUtils::getListKeypoints(keypoints, list, kps);
            Mat m = VOCUtils::drawKeyPoints(frame, kps, c.red, -1);
            // print the choice images
            String imName = "choice_cluster_";
            imName += std::to_string(*k).c_str();

            if(*k != 0)
            {
                String windowName = "Choose ";
                windowName += std::to_string(*k).c_str();
                windowName += "?";
                VOCUtils::display(windowName.c_str(), m);

                // Listen for a key pressed
                char c = ' ';
                while(true)
                {
                    if (c == 'n')   // next cluster
                    {

                        break;
                    }
                    else if (c == 'q')    // stop preview
                    {
                        break;
                    }
                    c = (char) waitKey(20);
                }
                destroyWindow(windowName.c_str());
                if(c == 'q')
                {
                    break;
                }
            }
        }

        cout << "You previewed minPts = " << sel << ". Do you want to use it? (yes/no): ";
        cin >> choiceStr;

        if(choiceStr.compare("yes") == 0 || choiceStr.compare("Yes") == 0 || choiceStr.compare("YES") == 0)
        {
            cout << "Chosen " << sel << endl;
            chosen = sel;
            done = true;
        }
    }
    cout << "-------------------------------------------------------------------------------" << endl;
    return chosen;
}

int main(int argc, char** argv)
{
    ocl::setUseOpenCL(true);
    Mat frame;
    VOCounter vcount;
    VideoCapture cap;
    Ptr<Feature2D> detector = SURF::create(MIN_HESSIAN);
    vsettings& settings = vcount.getSettings();

    cv::CommandLineParser parser(argc, argv,
                                 "{help ||}{o||}{n|1|}{R||}"
                                 "{v||}{video||}{image||}{w|1|}{s||}"
                                 "{c||}{t||}{l||}{tr|BOOSTING|}{tp||}"
                                 "{d||}{f||}{cm||}{I||}{co||}{ct||}"
                                 "{rx||}{ry||}{rw||}{rh||}{z||}"
                                 "{r||}{D||}{O||}{minPts|3|}"
                                 "{eo||}{fi||}{pa||}");

    if(!processOptions(settings, parser))
    {
        help();
        return -1;
    }

    if(!settings.inputFile.empty() && videoMode)
    {
        cap.open(settings.inputFile);

        if( !cap.isOpened() )
        {
            printf("Could not open stream\n");
            return -1;
        }
    }    

    // process the vcount settings to create necessary output folders
    vcount.processSettings();

    bool tmp = true;

    while(tmp)
    {
        if(videoMode)
        {
            tmp = cap.read(frame);
        } else {
            frame = imread(settings.inputFile);
            tmp = false;
        }

        if(!tmp && videoMode)
        {
            continue;
        }

        vector<KeyPoint> keypoints;
        Mat descriptors;
        detector->detectAndCompute(frame, UMat(), keypoints, descriptors);

        /**
         * Finding the colour model for the current frame
         */
        if(vcount.getFrameCount() == 0 && settings.colourModelTracking)
        {
            cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Detecting Colour Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
            printf("Finding proper value of minPts\n");
            vcount.trainColourModel(frame, keypoints);
            int32_t chosen = consolePreviewColours(frame, keypoints, vcount.getColourModelMaps(), vcount.getValidities(), vcount.getColourModel().getMinPts());
            vcount.getLearnedColourModel(chosen);
            cout << "Selected ............ " << vcount.getColourModel().getMinPts() << endl;
            vcount.chooseColourModel(frame, descriptors, keypoints);
        }

        // Listen for a key pressed
        char c = (char) waitKey(20);
        if (c == 'q')
        {
            break;
        }
        else if (c == 's')     // select a roi if c has een pressed or if the program was run with -s option
        {
            settings.selectROI = true;
        }
        vcount.processFrame(frame, descriptors, keypoints);
    }

    return 0;
}


