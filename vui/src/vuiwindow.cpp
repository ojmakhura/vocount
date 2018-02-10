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
