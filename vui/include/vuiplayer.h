#ifndef VUIPLAYER_H
#define VUIPLAYER_H

#include <QMutex>
#include <QThread>
#include <QImage>
#include <QWaitCondition>

namespace Ui {
class VUIPlayer;
}

class VUIPlayer : public QThread
{

protected:
    void run();
    void msleep(int ms);

public:
    VUIPlayer();
    VUIPlayer(QObject *parent = 0);
    //Destructor
    ~VUIPlayer();
    //Load a video from memory
    bool loadVideo(QString filename);
    void play();
    //Stop the video
    void stop();
    //check if the player has been stopped
    bool isStopped() const;
    void pause();
    void resume();
    bool isPaused();
};

#endif // VUIPLAYER_H
