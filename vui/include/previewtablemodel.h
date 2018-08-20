#ifndef PREVIEWTABLEMODEL_H
#define PREVIEWTABLEMODEL_H

#include <QAbstractTableModel>

class PreviewTableModel : public QAbstractTableModel
{
    Q_OBJECT

public:
    explicit PreviewTableModel(QObject *parent = nullptr);

    // Header:
    QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;

    // Basic functionality:
    int rowCount(const QModelIndex &parent = QModelIndex()) const override;
    int columnCount(const QModelIndex &parent = QModelIndex()) const override;

    QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;

private:
};

#endif // PREVIEWTABLEMODEL_H