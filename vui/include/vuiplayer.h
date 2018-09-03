#ifndef VUIPLAYER_H
#define VUIPLAYER_H

#include <QMutex>
#include <QThread>
#include <QImage>
#include <QWaitCondition>
#include <opencv/cv.hpp>
#include "vocount/vocounter.hpp"

using namespace cv;
using namespace vocount;

class VUIPlayer : public QThread
{

protected:
    void run();
    void msleep(int ms);

public:
    vsettings settings;
    bool initialised = false;
    vocount vcount;
    
    explicit VUIPlayer(QObject *parent = 0);
    //Destructor
    ~VUIPlayer();
    void play();
    //Stop the video
    void stop();
    //check if the player has been stopped
    bool isStopped() const;
    void pause();
    void resume();
    bool isPaused();
    void initPlayer();

private:
	Mat frame;
    VOCounter vcount;
    VideoCapture cap;
    Ptr<Feature2D> detector;
    bool _paused = false;
    bool _stop = false;
    QMutex mutex;
    QWaitCondition condition;

};

#endif // VUIPLAYER_H
