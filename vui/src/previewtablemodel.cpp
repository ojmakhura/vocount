#include "previewtablemodel.h"

PreviewTableModel::PreviewTableModel(QObject *parent)
    : QAbstractTableModel(parent)
{
}

PreviewTableModel::PreviewTableModel(QList<ClusterPreviewItem> items, QObject *parent)
    : QAbstractTableModel(parent)
    , items(items)
{

}

QVariant PreviewTableModel::headerData(int section, Qt::Orientation orientation, int role) const
{
    if (role != Qt::DisplayRole)
            return QVariant();

    if (orientation == Qt::Horizontal) {
        switch (section) {
            case 0:
                return tr("minPts");

            case 1:
                return tr("Num. of Clusters");

            case 2:
                return tr("Validity");

            case 3:
                return tr("Preview");

            default:
                return QVariant();
        }
    }
    return QVariant();
}

int PreviewTableModel::rowCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return 0;

    return items.size();
}

int PreviewTableModel::columnCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return 0;

    return 4;
}

QVariant PreviewTableModel::data(const QModelIndex &index, int role) const
{
    if (!index.isValid())
        return QVariant();

    if (index.row() >= items.size() || index.row() < 0)
           return QVariant();

    if(role == Qt::DisplayRole){
        const auto &item = items.at(index.row());

        if (index.column() == 0)
            return item.getMinPts();
        else if (index.column() == 1)
            return item.getNumberOfClusters();
        else if (index.column() == 2)
            return item.getValidity();
        else if (index.column() == 3)
            return "QButton";

    }

    return QVariant();
}

QList<ClusterPreviewItem> PreviewTableModel::getItems() const
{
    return items;
}

void PreviewTableModel::setItems(const QList<ClusterPreviewItem> &value)
{
    items = value;
}
