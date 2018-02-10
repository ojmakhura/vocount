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

private:
    Ui::VUIWindow *ui;
};

#endif // VUIWINDOW_H
