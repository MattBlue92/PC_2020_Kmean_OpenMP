//
// Created by matteo on 18/08/20.
//

#ifndef PC_KMEANS_KMEANS_H
#define PC_KMEANS_KMEANS_H


#include <math.h>
#include <float.h>
#include <iostream>
#include <cstdlib>

using namespace std;

class Kmeans {

private:
    double **centroids = nullptr;
    int *labels = nullptr;
    double tol = 0.0001;
    double SSE = 0;
    int it = 100;
    int seed = 0;
    int k = 2;

    void centroidUpdate(int *labels, double **centroids, double **dataset, int nCol, int nRow);

    int clusterAssignment(double *point, double **centroids, int nCol);

    double getDistance(double *point, double *centroid, int nCol);

    void initCentroids(double **dataMatrix, int nRow, int nCol);

    void saveCentroid(double **save, double **centroid, int dim);

    void statistics(double SSE, double iterations, double **centroids, int nCol);

    void printMatrix(double **matrix, int nRow, int nCol);

    double computeSSE(int nCol, double *const *centroids, double *const *oldCentroids);

public:
    Kmeans(int k);

    Kmeans(int k, int it, int seed, double tol);

    ~Kmeans();//metodo distruttore

    int *fitPredict(double **dataset, int nRow, int nCol);

    void restart(int k);

    void restart(int k, int it, int seed, double tol);

    int *getLabels();

    double **clusterCenters();

    double getSSE();

};


#endif //PC_KMEANS_KMEANS_H

