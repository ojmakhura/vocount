#include "vuiwindow.h"
#include "ui_vuiwindow.h"

VUIWindow::VUIWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::VUIWindow)
{
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

}

void VUIWindow::on_outputFolderButton_clicked()
{

}

void VUIWindow::on_truthFolderButton_clicked()
{

}
