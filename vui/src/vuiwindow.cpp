#include "vuiwindow.h"
#include "ui_vuiwindow.h"

#include <QMainWindow>
#include <QFileDialog>
#include <QMessageBox>

//using namespace Ui;

VUIWindow::VUIWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::VUIWindow)
{
    player = new VUIPlayer();
    ui->setupUi(this);
}

VUIWindow::~VUIWindow()
{
	delete player;
    delete ui;
}


void VUIWindow::on_videoSelectButton_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Open Video"), ".", tr("Video Files (*.avi *.mpg *.mp4)"));

    if(!filename.isEmpty()){        
        ui->actionPlay->setEnabled(true);
        this->player->settings.inputVideo = filename.toStdString().data();
        ui->videoSelectEdit->setText(filename);
    }
}

void VUIWindow::on_outputFolderButton_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("Select Output Folder"),
                                                    ".",
                                                    QFileDialog::ShowDirsOnly
                                                    | QFileDialog::DontResolveSymlinks);
    if(!dir.isEmpty()){
        ui->outFolderSelectEdit->setText(dir);
        this->player->settings.outputFolder = dir.toStdString().data();
        this->player->settings.print = true;
    }
}

void VUIWindow::on_truthFolderButton_clicked()
{
    QString dir = QFileDialog::getExistingDirectory(this, tr("Select Truth Folder"),
                                                    ".",
                                                    QFileDialog::ShowDirsOnly
                                                    | QFileDialog::DontResolveSymlinks);
    if(!dir.isEmpty()){
        ui->truthFolderSelectEdit->setText(dir);
        this->player->settings.truthFolder = dir.toStdString().data();
    }
}

void VUIWindow::on_actionPlay_triggered()
{
    if(!this->player->initialised){
        this->player->initPlayer();
        this->player->initialised = true;
        ui->actionStop->setEnabled(true);
    }

    ui->actionPause->setEnabled(true);
    ui->actionPlay->setEnabled(false);
    player->play();

}

void VUIWindow::on_actionStop_triggered()
{
    ui->actionPause->setEnabled(false);
    ui->actionPlay->setEnabled(true);
    ui->actionStop->setEnabled(false);
    player->stop();
}

void VUIWindow::on_actionExit_triggered()
{
    exit(0);
}

void VUIWindow::on_actionPause_triggered()
{
    ui->actionPlay->setEnabled(true);
    ui->actionPause->setEnabled(false);
    player->pause();
}
