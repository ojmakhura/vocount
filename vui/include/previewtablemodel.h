#ifndef PREVIEWTABLEMODEL_H
#define PREVIEWTABLEMODEL_H

#include <QAbstractTableModel>
#include "clusterpreviewitem.h"

class PreviewTableModel : public QAbstractTableModel
{
    Q_OBJECT

public:
    explicit PreviewTableModel(QObject *parent = nullptr);
    PreviewTableModel(QList<ClusterPreviewItem> items, QObject *parent = nullptr);

    // Header:
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;

    // Basic functionality:
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    int columnCount(const QModelIndex &parent = QModelIndex()) const override;

    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;

    QList<ClusterPreviewItem> getItems() const;
    void setItems(const QList<ClusterPreviewItem> &value);

private:
    QList<ClusterPreviewItem> items;
};

#endif // PREVIEWTABLEMODEL_H
