#include "previewdialog.h"
#include "ui_previewdialog.h"

PreviewDialog::PreviewDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::PreviewDialog)
{
    ui->setupUi(this);
    //tableModel = new PreviewTableModel();
    //ui->tableView->setModel(tableModel);
}

PreviewDialog::PreviewDialog(QWidget *parent, map<int, IntIntListMap* >* clusterMaps, vector<int32_t>& validities) :
    QDialog(parent),
    ui(new Ui::PreviewDialog)
{

    QList<ClusterPreviewItem> items;

    for(map<int, IntDoubleListMap* >::iterator it = clusterMaps->begin(); it != clusterMaps->end(); ++it){
        ClusterPreviewItem item(it->first, g_hash_table_size(it->second), validities[it->first - 3] );
        items.push_back(item);
    }

    tableModel = new PreviewTableModel(items);
    ui->tableView->setModel(tableModel);
    ui->setupUi(this);
}

PreviewDialog::~PreviewDialog()
{
    delete tableModel;
    delete ui;

}

void PreviewDialog::on_pushButton_clicked()
{
    this->hide();
}
