#include "vuiplayer.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <ctime>
#include <string>
#include "vocount/print_utils.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

VUIPlayer::VUIPlayer(QObject *parent)
    : QThread(parent){
    //ocl::setUseOpenCL(true);
    this->settings.isConsole = true;
    this->vcount.colourSel.minPts = -1;
	this->vcount.detector = SURF::create(MIN_HESSIAN);
}

VUIPlayer::~VUIPlayer(){    
    mutex.lock();
    _stop = true;
    cap.release();
    condition.wakeOne();
    mutex.unlock();
    wait();
}

void VUIPlayer::initPlayer(){
    if(this->settings.print){
        createOutputDirectories(vcount, settings);
        printf("Output directories created\n");
    }

    if(!this->cap.open(settings.inputVideo)){
        printf("Could not open video %s\n", settings.inputVideo.c_str());
    } else{
        printf("Video successfully opened.\n");
    }

    if(!this->settings.truthFolder.empty()){
        getFrameTruth(settings.truthFolder, vcount.truth);
    }

    settings.rsize = 1;
    settings.selectROI = true;
}

void VUIPlayer::run(){
    while(!_stop){
        if(!_paused){
            if(!this->cap.read(this->frame)){
                _stop = true;
                continue;
            }

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
                vector<int32_t> validities = trainColourModel(this->vcount.colourSel, frame, f.keypoints, clusterMaps, vcount.trainingFile, settings.isConsole);
                getLearnedColourModel(vcount.colourSel, clusterMaps, validities);
                chooseColourModel(frame, f.descriptors, f.keypoints, vcount.colourSel);
                if(vcount.trackingFile.is_open()){
                    vcount.trackingFile << 1 << "," << f.keypoints.size() << "," << vcount.colourSel.selectedDesc.rows << "," << vcount.colourSel.minPts << "," << vcount.colourSel.numClusters << "," << vcount.colourSel.validity << endl;
                }
            }
            processFrame(vcount, settings, f, frame);
        }
        this->msleep(0);
    }
    stop();
}

void VUIPlayer::play(){
    _paused = false;
    if(!isRunning()){
        start(LowPriority);
    }
}

void VUIPlayer::msleep(int ms){
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000 * 1000 };
    nanosleep(&ts, NULL);
}

void VUIPlayer::pause(){
    _paused = true;
}

void VUIPlayer::resume(){
    _paused = false;
}

bool VUIPlayer::isPaused(){
    return _paused;
}

void VUIPlayer::stop(){
    this->initialised = false;
    this->cap.release();
    finalise(vcount);
    _stop = true;
}

bool VUIPlayer::isStopped() const{
    return _stop;
}
