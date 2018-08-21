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

    ///dialog = new PreviewDialog(this);
    ui->trackerComboBox->addItem("BOOSTING");
    ui->trackerComboBox->addItem("KCF");
    ui->trackerComboBox->addItem("TLD");
    ui->trackerComboBox->addItem("MEDIANFLOW");
    ui->trackerComboBox->addItem("GOTURN");
    ui->trackerComboBox->addItem("MIL");
}

VUIWindow::~VUIWindow()
{
    delete dialog;
	delete player;
    delete ui;
}


void VUIWindow::on_videoSelectButton_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, tr("Open Video"), ".", tr("Video Files (*.avi *.mpg *.mp4)"));

    if(!filename.isEmpty()){        
        ui->actionPlay->setEnabled(true);
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
    }
}

void VUIWindow::on_actionPlay_triggered()
{
    if(!this->player->initialised){
        this->player->initialised = true;
        ui->actionStop->setEnabled(true);

        // player settings
        this->player->settings.inputVideo = ui->videoSelectEdit->text().toStdString().c_str();

        if(!ui->outFolderSelectEdit->text().isEmpty()){
            this->player->settings.outputFolder = ui->outFolderSelectEdit->text().toStdString().c_str();
            this->player->settings.print = true;
        }

        if(!ui->truthFolderSelectEdit->text().isEmpty()){
            this->player->settings.truthFolder = ui->truthFolderSelectEdit->text().toStdString().c_str();

        }

        if(!ui->sampleSizeEdit->text().isEmpty()){
            player->settings.rsize = ui->sampleSizeEdit->text().toInt();
        } else{
            player->settings.rsize = 1;
        }

        player->vcount.roiExtracted = false;

        if(!ui->xLineEdit->text().isEmpty()){
            player->vcount.roi.x = ui->xLineEdit->text().toInt();
            player->vcount.roiExtracted = true;
        }

        if(player->vcount.roiExtracted && !ui->yLineEdit->text().isEmpty()){
            player->vcount.roi.y = ui->yLineEdit->text().toInt();
        } else {
            player->vcount.roiExtracted = false;
        }

        if(player->vcount.roiExtracted && !ui->widthLineEdit->text().isEmpty()){
            player->vcount.roi.width = ui->widthLineEdit->text().toInt();
        } else {
            player->vcount.roiExtracted = false;
        }

        if(player->vcount.roiExtracted && !ui->heightLineEdit->text().isEmpty()){
            player->vcount.roi.height = ui->heightLineEdit->text().toInt();
        } else {
            player->vcount.roiExtracted = false;
        }

        if(player->vcount.roiExtracted){
            player->settings.selectROI = false;
        } else {
            player->vcount.roi = Rect2d(0, 0, 0, 0);
        }

        player->settings.trackerAlgorithm = ui->trackerComboBox->currentText().toStdString().data();
        ui->trackerComboBox->setEnabled(false);
        player->settings.dClustering = ui->descriptorSpaceBox->isChecked();
        player->settings.fdClustering = ui->filteredDescriptorBox->isChecked();
        player->settings.dfClustering = ui->descFilteredDescBox->isChecked();
        player->settings.extend = ui->extendCheckBox->isChecked();
        player->settings.rotationalInvariance = ui->rotationInvariance->isChecked();
        player->settings.step = 1;

        this->player->initPlayer();
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

void VUIWindow::on_descFilteredDescBox_clicked()
{
    if(ui->descFilteredDescBox->isChecked()){
        ui->descriptorSpaceBox->setChecked(true);
        ui->filteredDescriptorBox->setChecked(true);
    } else{
        ui->descriptorSpaceBox->setChecked(false);
        ui->filteredDescriptorBox->setChecked(false);
    }
}


