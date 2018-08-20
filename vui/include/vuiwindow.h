#ifndef VUIWINDOW_H
#define VUIWINDOW_H

#include <QMainWindow>

#include "vuiplayer.h"
#include "previewdialog.h"

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

    void on_descFilteredDescBox_clicked();

    void on_descImageSpaceBox_clicked();

    void on_combineAllBox_clicked();

private:
    Ui::VUIWindow *ui;
    VUIPlayer *player;
    PreviewDialog *dialog;

};

#endif // VUIWINDOW_H
