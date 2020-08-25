//
// Created by Matteo Marulli and Matteo Gemignani  on 21/08/20.
//

#include "KmeansOpenMP.h"


void KmeansOpenMP::initCentroids(double **dataMatrix, int nRow, int nCol) {
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

double KmeansOpenMP::getDistance(double *point, double *centroid, int nCol) {
    double distance = 0;
    for (int i = 0; i < nCol; i++) {
        distance = distance + pow(point[i] - centroid[i], 2);
    }
    return sqrt(distance);
}


int KmeansOpenMP::clusterAssignment(double *point, double **centroids, int nCol) {
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

void
KmeansOpenMP::centroidUpdate(int *labels, int *labelsCount, double **centroids, double **dataset, int nCol, int nRow) {
    int pointForCluster;
#pragma omp parallel default(none) shared(centroids, labelsCount) firstprivate (nCol, nRow, labels, dataset)
    {
#pragma omp for collapse(2)
        for (int c = 0; c < k; c++) {
            for (int i = 0; i < nCol; i++) {
                centroids[c][i] = 0;
            }
        }
#pragma omp for
        for (int i = 0; i < k; i++) {
            labelsCount[i] = 0;
        }

#pragma omp for
        for (int c = 0; c < k; c++) {
            for (int j = 0; j < nRow; j++) {
                if (c == labels[j]) {
                    labelsCount[c]++;
                    for (int i = 0; i < nCol; i++) {
                        centroids[c][i] += dataset[j][i];
                    }

                }
            }
        }
#pragma omp for collapse(2)
        for (int c = 0; c < k; c++) {
            for (int i = 0; i < nCol; i++) {
                centroids[c][i] =
                        centroids[c][i] / labelsCount[c]; // dividi ogni componente per il numero di oggetti nel cluster
            }
        }
    }
}

void KmeansOpenMP::saveCentroid(double **save, double **centroid, int dim) {
#pragma omp parallel for collapse(2) shared(save) firstprivate(centroid, dim)
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < dim; j++) {
            save[i][j] = centroid[i][j];
        }
    }
}

double KmeansOpenMP::computeSSE(int nCol, double *const *centroids, double *const *oldCentroids) {
    double sum = 0;
#pragma omp parallel for reduction(+:sum)
    for (int c = 0; c < k; c++) {
        sum += pow(getDistance(oldCentroids[c], centroids[c], nCol), 2);
    }
    return sum;
}

int *KmeansOpenMP::fitPredict(double **dataset, int nRow, int nCol) {
    int t = 0;
    double SSE = 0;
    int *labels = new int[nRow];

    int *labelsCount = new int[k];


    initCentroids(dataset, nRow, nCol);

    double **oldCentroids;
    oldCentroids = new double *[k];
    for (int i = 0; i < k; i++) {
        oldCentroids[i] = new double[nCol];
    }

    do {
        t++;
#pragma omp parallel for schedule(static, nRow/10) default(none) shared(labels) firstprivate(centroids, nCol, nRow, dataset)
        for (int indexPoint = 0; indexPoint < nRow; indexPoint++) {
            labels[indexPoint] = clusterAssignment(dataset[indexPoint], centroids, nCol);
        }

        saveCentroid(oldCentroids, centroids, nCol);
        centroidUpdate(labels, labelsCount, centroids, dataset, nCol, nRow);


        SSE = computeSSE(nCol, centroids, oldCentroids);
        //cout<<t<<"/"<<it<<" SSE:"<<SSE<<endl;
    } while (SSE > tol and t < it);
    statistics(SSE, t, centroids, nCol);
    //destroy obj with *
    delete[] oldCentroids;

    return labels;
}

void KmeansOpenMP::printMatrix(double **matrix, int nRow, int nCol) {
    for (int i = 0; i < nRow; i++) {
        cout << "Centroid " << i << " : [";
        for (int j = 0; j < nCol; j++)
            cout << matrix[i][j] << ", ";
        cout << "]" << endl;
    }
}

void KmeansOpenMP::statistics(double SSE, double iterations, double **centroids, int nCol) {
    cout << "<<<<<<Statistics<<<<<<" << endl;
    cout << "SSE: " << SSE << endl;
    cout << "Iterations: " << iterations << endl;
    cout << "Centroids: " << endl;
    printMatrix(centroids, k, nCol);
    cout << "<<<<<<End-Statistics<<<<<<" << endl;
}

void KmeansOpenMP::restart(int k) {
    centroids = nullptr;
    labels = nullptr;
    tol = 0.0001;
    SSE = 0;
    it = 10;
    this->k = k;
}

double KmeansOpenMP::getSSE() {
    return SSE;
}

double **KmeansOpenMP::clusterCenters() {
    return centroids;
}

int *KmeansOpenMP::getLabels() {
    return labels;
}

KmeansOpenMP::KmeansOpenMP(int k) {
    this->k = k;
}

KmeansOpenMP::KmeansOpenMP(int k, int it, int seed, double tol) {
    this->k = k;
    this->it = it;
    this->seed = seed;
    this->tol = tol;
}

void KmeansOpenMP::restart(int k, int it, int seed, double tol) {
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

KmeansOpenMP::~KmeansOpenMP() {
    delete[] centroids;
    delete[] labels;
}


