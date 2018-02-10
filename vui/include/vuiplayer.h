#ifndef VUIPLAYER_H
#define VUIPLAYER_H

#include <QMutex>
#include <QThread>
#include <QImage>
#include <QWaitCondition>

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
    bool loadVideo(String filename);
    void Play();
    //Stop the video
    void Stop();
    //check if the player has been stopped
    bool isStopped() const;
    void pause();
    void resume();
    bool isPaused();
};

#endif // VUIPLAYER_H
