#include "vuiplayer.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <opencv2/core/ocl.hpp>
#include <opencv2/xfeatures2d.hpp>
//#include <opencv2/ximgproc/segmentation.hpp>
#include <ctime>
#include <string>
#include "vocount/print_utils.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

VUIPlayer::VUIPlayer(QObject *parent)
    : QThread(parent){
    //ocl::setUseOpenCL(true);
    this->colourSel.minPts = -1;
	this->vcount.detector = SURF::create(1000);
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

    if(this->settings.truthFolder.length() > 0){
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
            //display("Frame", this->frame);
            processFrame(vcount, settings, colourSel, frame);
        }
        this->msleep(0);
    }
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
    _stop = true;
}

bool VUIPlayer::isStopped() const{
    return _stop;
}
