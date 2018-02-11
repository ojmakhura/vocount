#ifndef VUIWINDOW_H
#define VUIWINDOW_H

#include <QMainWindow>

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
    void on_exitButton_clicked();

    void on_stopButton_clicked();

    void on_runButton_clicked();

    void on_videoSelectButton_clicked();

    void on_outputFolderButton_clicked();

    void on_truthFolderButton_clicked();

private:
    Ui::VUIWindow *ui;
};

#endif // VUIWINDOW_H
