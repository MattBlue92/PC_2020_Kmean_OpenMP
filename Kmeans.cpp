//
// Created by Matteo Marulli and Matteo Gemignani on 18/08/20.
//

#include "Kmeans.h"


void Kmeans::initCentroids(double **dataMatrix, int nRow, int nCol) {
    centroids = new double *[k];
    for (int i = 0; i < k; i++) {
        centroids[i] = new double[nCol];
    }
    srand(seed);
    for (int i = 0; i < k; i++) {
        int row = int(rand() % nRow);
        for (int j = 0; j < nCol; j++) {
            centroids[i][j] = dataMatrix[row][j];
        }
    }

}


double Kmeans::getDistance(double *point, double *centroid, int nCol) {
    double distance = 0;
    for (int i = 0; i < nCol; i++) {
        distance = distance + pow(point[i] - centroid[i], 2);
    }
    return sqrt(distance);
}

int Kmeans::clusterAssignment(double *point, double **centroids, int nCol) {
    int index = 0;
    double distance = 0;
    double oldDistance = DBL_MAX;
    int dim = sizeof(point) / sizeof(point[0]);
    for (int i = 0; i < k; i++) {
        double *centroid_i = centroids[i];
        distance = getDistance(point, centroid_i, dim);
        if (distance < oldDistance) {
            index = i;
            oldDistance = distance;
        }
    }
    return index;
}


void Kmeans::centroidUpdate(int *labels, double **centroids, double **dataset, int nCol, int nRow) {

    for (int c = 0; c < k; c++) {
        int pointForCluster = 0;

        for (int i = 0; i < nCol; i++) {
            centroids[c][i] = 0;
        }

        for (int j = 0; j < nRow; j++) {
            if (c == labels[j]) {
                pointForCluster++;
                for (int i = 0; i < nCol; i++) {
                    centroids[c][i] = centroids[c][i] + dataset[j][i];
                }
            }
        }

        for (int i = 0; i < nCol; i++) {
            centroids[c][i] = centroids[c][i] / pointForCluster;
        }


    }

}


void Kmeans::saveCentroid(double **save, double **centroid, int dim) {
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < dim; j++) {
            save[i][j] = centroid[i][j];
        }
    }
}

int *Kmeans::fitPredict(double **dataset, int nRow, int nCol) {
    int t = 0;
    double SSE = 0;
    int *labels = new int[nRow];

    initCentroids(dataset, nRow, nCol);
    double **oldCentroids;
    oldCentroids = new double *[k];
    for (int i = 0; i < k; i++) {
        oldCentroids[i] = new double[nCol];
    }

    do {
        t++;

        for (int indexPoint = 0; indexPoint < nRow; indexPoint++) {
            labels[indexPoint] = clusterAssignment(dataset[indexPoint], centroids, nCol);
        }

        saveCentroid(oldCentroids, centroids, nCol);
        centroidUpdate(labels, centroids, dataset, nCol, nRow);

        SSE = computeSSE(nCol, centroids, oldCentroids);
        //cout << t << "/" << it << " SSE:" << SSE << endl;
    } while (SSE > tol and t < it);

    statistics(SSE, t, centroids, nCol);
    //destroy obj with *
    delete[] oldCentroids;

    return labels;
}

double Kmeans::computeSSE(int nCol, double *const *centroids, double *const *oldCentroids) {
    double sum = 0;
    for (int c = 0; c < k; c++) {
        sum = sum + pow(getDistance(oldCentroids[c], centroids[c], nCol), 2);
    }
    return sum;
}


void Kmeans::printMatrix(double **matrix, int nRow, int nCol) {
    for (int i = 0; i < nRow; i++) {
        cout << "Centroid " << i << " : [";
        for (int j = 0; j < nCol; j++)
            cout << matrix[i][j] << ", ";
        cout << "]" << endl;
    }
}

void Kmeans::statistics(double SSE, double iterations, double **centroids, int nCol) {
    cout << "<<<<<<Statistics<<<<<<" << endl;
    cout << "SSE: " << SSE << endl;
    cout << "Iterations: " << iterations << endl;
    cout << "Centroids: " << endl;
    printMatrix(centroids, k, nCol);
    cout << "<<<<<<End-Statistics<<<<<<" << endl;
}

void Kmeans::restart(int k) {
    centroids = nullptr;
    labels = nullptr;
    tol = 0.0001;
    SSE = 0;
    it = 100;
    this->k = k;
}

double Kmeans::getSSE() {
    return SSE;
}

double **Kmeans::clusterCenters() {
    return centroids;
}

int *Kmeans::getLabels() {
    return labels;
}

Kmeans::Kmeans(int k) {
    this->k = k;
}

Kmeans::Kmeans(int k, int it, int seed, double tol) {
    this->k = k;
    this->it = it;
    this->seed = seed;
    this->tol = tol;
}

void Kmeans::restart(int k, int it, int seed, double tol) {
    delete[] centroids;
    delete[] labels;
    centroids = nullptr;
    labels = nullptr;
    SSE = 0;
    this->tol = tol;
    this->it = it;
    this->seed = seed;
    this->k = k;
}

Kmeans::~Kmeans() {
    delete[] centroids;
    delete[] labels;
}
