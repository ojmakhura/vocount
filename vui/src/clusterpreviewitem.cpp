#include "include/clusterpreviewitem.h"

ClusterPreviewItem::ClusterPreviewItem()
{

}

ClusterPreviewItem::ClusterPreviewItem(int minPts, int numberOfClusters, int validity){
    this->minPts = minPts;
    this->numberOfClusters = numberOfClusters;
    this->validity = validity;
}

int ClusterPreviewItem::getMinPts() const
{
    return minPts;
}

void ClusterPreviewItem::setMinPts(int value)
{
    minPts = value;
}

int ClusterPreviewItem::getNumberOfClusters() const
{
    return numberOfClusters;
}

void ClusterPreviewItem::setNumberOfClusters(int value)
{
    numberOfClusters = value;
}

int ClusterPreviewItem::getValidity() const
{
    return validity;
}

void ClusterPreviewItem::setValidity(int value)
{
    validity = value;
}
