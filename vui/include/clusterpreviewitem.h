#ifndef CLUSTERPREVIEWITEM_H
#define CLUSTERPREVIEWITEM_H


class ClusterPreviewItem
{
public:
    ClusterPreviewItem();
    ClusterPreviewItem(int minPts, int numberOfClusters, int validity);

    int getMinPts() const;
    void setMinPts(int value);

    int getNumberOfClusters() const;
    void setNumberOfClusters(int value);

    int getValidity() const;
    void setValidity(int value);

private:
    int minPts;
    int numberOfClusters;
    int validity;
};

#endif // CLUSTERPREVIEWITEM_H
