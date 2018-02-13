#ifndef VUIWINDOW_H
#define VUIWINDOW_H

#include <QMainWindow>

#include "vuiplayer.h"

namespace Ui {
class VUIWindow;
}

class VUIWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit VUIWindow(QWidget *parent = 0);
    ~VUIWindow();

private slots:
    void on_videoSelectButton_clicked();

    void on_outputFolderButton_clicked();

    void on_truthFolderButton_clicked();

    void on_actionPlay_triggered();

    void on_actionStop_triggered();

    void on_actionExit_triggered();

    void on_actionPause_triggered();

private:
    Ui::VUIWindow *ui;
    VUIPlayer *player;
};

#endif // VUIWINDOW_H
