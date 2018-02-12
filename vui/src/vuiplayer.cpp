#include "vuiplayer.h"
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
#include "vocount/print_utils.hpp"

VUIPlayer::VUIPlayer()
{
    ocl::setUseOpenCL(true);
}

VUIPlayer::VUIPlayer(QObject *parent)
    : QThread(parent){
    ocl::setUseOpenCL(true);
}

VUIPlayer::~VUIPlayer(){

}

void VUIPlayer::run(){

}

void VUIPlayer::msleep(int ms){
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000 * 1000 };
    nanosleep(&ts, NULL);
}

void VUIPlayer::loadVideo(QString filename){

}
