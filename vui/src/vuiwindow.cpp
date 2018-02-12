#include "vuiwindow.h"
#include "ui_vuiwindow.h"

#include <QMainWindow>
#include <QFileDialog>
#include <QMessageBox>

VUIWindow::VUIWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::VUIWindow)
{
    player = new VUIPlayer();
    ui->setupUi(this);
}

VUIWindow::~VUIWindow()
{
    delete ui;
}

void VUIWindow::on_exitButton_clicked()
{
    exit(0);
}

void VUIWindow::on_stopButton_clicked()
{

}

void VUIWindow::on_runButton_clicked()
{

}

void VUIWindow::on_videoSelectButton_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Open Video"), ".", tr("Video Files (*.avi *.mpg *.mp4)"));

    if(!filename.isEmpty()){


        ui->actionLoad_Video->setEnabled(false);
        if(!player->loadVideo(filename.toStdString().data())){
            QMessageBox msgBox;
            msgBox.setText("Video could not be opened.");
            msgBox.exec();
        }
        ui->runButton->setEnabled(true);
    }
}

void VUIWindow::on_outputFolderButton_clicked()
{

}

void VUIWindow::on_truthFolderButton_clicked()
{

}
