#include "vuiplayer.h"

VUIPlayer::VUIPlayer()
{

}

VUIPlayer::VUIPlayer(QObject *parent)
    : QThread(parent){

}

VUIPlayer::~VUIPlayer(){

}

void VUIPlayer::run(){

}

void VUIPlayer::msleep(int ms){
    struct timespec ts = { ms / 1000, (ms % 1000) * 1000 * 1000 };
    nanosleep(&ts, NULL);
}
