#ifndef PREVIEWDIALOG_H
#define PREVIEWDIALOG_H

#include <QDialog>
#include "previewtablemodel.h"
#include "vocount/process_frame.hpp"

namespace Ui {
class PreviewDialog;
}

class PreviewDialog : public QDialog
{
    Q_OBJECT

public:
    explicit PreviewDialog(QWidget *parent = 0);
    PreviewDialog(QWidget *parent, map<int, IntIntListMap* >* clusterMaps, vector<int32_t>& validities);
    ~PreviewDialog();

private slots:
    void on_pushButton_clicked();

private:
    Ui::PreviewDialog *ui;
    PreviewTableModel *tableModel;
};

#endif // PREVIEWDIALOG_H
