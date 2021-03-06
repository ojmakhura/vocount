#include "vuiplayer.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <ctime>
#include <string>
#include "vocount/voprinter.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

VUIPlayer::VUIPlayer(QObject *parent)
    : QThread(parent){
    //ocl::setUseOpenCL(true);
    this->detector = SURF::create(MIN_HESSIAN);
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
            vector<KeyPoint> keypoints;
            Mat descriptors;
            detector->detectAndCompute(frame, Mat(), keypoints, descriptors);

            /**
             * Finding the colour model for the current frame
             */
            if(vcount.getFrameCount() == 0 && settings.fdClustering)
            {
                cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Detecting Colour Model ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << endl;
                printf("Finding proper value of minPts\n");
                vcount.trainColourModel(frame, keypoints);
                //int32_t chosen = consolePreviewColours(frame, keypoints, vcount.getColourModelMaps(), vcount.getValidities(), vcount.getColourModel()->getMinPts());
                vcount.getLearnedColourModel(vcount.getColourModel()->getMinPts());
                cout << " Selected ............ " << vcount.getColourModel()->getMinPts() << endl;
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
    //finalise(vcount);
    _stop = true;
}

bool VUIPlayer::isStopped() const{
    return _stop;
}
