// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include <random>
#include <fstream>  
#define MAX_DISPARITY 60
#define SMALL_PENALTY 10
#define BIG_PENALTY 30

using namespace std;

char fname[250];
Mat resizedImgLeft, resizedImgRight;
int correspondingPixelCount[MAX_DISPARITY];
double*** mutualProbabilityDistribution;
double** individualProbabilityLeft;
double** individualProbabilityRight;
double gaussian2D[5][5] = { {1,4,7,4,1},{4,16,26,16,4},{7,26,41,26,7},{4,16,26,16,4},{1,4,7,4,1} };
double gaussian1D[7] = {0.006,0.061,0.242,0.383,0.242,0.061,0.006};
double** individualEntropyLeft;
double** individualEntropyRight;
double*** mutualEntropy;
double*** mutualInformation;
double*** paths, ***frequency, ***cost;

double*** costMatrix;

void populateGaussian2D(){
	for(int i = 0; i < 5; i++)
		for(int j = 0; j < 5; j++)
			gaussian2D[i][j] /= (273 * 1.0);
}
double*** allocate3D(int x, int y, int z) {
	double*** arr = (double***)calloc(x, sizeof(double**));
	for (int i = 0; i < x; i++) {
		arr[i] = (double**)calloc(y, sizeof(double*));
		for (int j = 0; j < y; j++) {
			arr[i][j] = (double*)calloc(z, sizeof(double));
		}
	}
	return arr;
}

double** allocate2D(int x, int y){
	double** arr = (double**)calloc(x, sizeof(double*));
	for(int i = 0; i<x;i++){
		arr[i] = (double*)calloc(y, sizeof(double));
	}
	return arr;
}

void free2D(double** arr, int x) {
	for (int i = 0; i < x; i++) {
		free(arr[i]);
	}
	free(arr);
}

void free3D(double*** arr, int x, int y) {
	for (int i = 0; i < x; i++) {
		for (int j = 0; j < y; j++) {
			free(arr[i][j]);
		}
		free(arr[i]);
	}
	free(arr);
}

bool isInside(int row, int column, int rowCount, int colCount) {
	return row >= 0 &&
		row < rowCount &&
		column >= 0 &&
		column < colCount;
}

uchar RGBtoGrayscale(Vec3b pixel) {
	uchar R = pixel[2];
	uchar G = pixel[1];
	uchar B = pixel[0];

	return R / 3.0 + G / 3.0 + B / 3.0;
}

//conversion of a colored image to grayscale
Mat convertToGrayscale(Mat image) {
	Mat result(image.rows, image.cols, CV_8UC1);

	for (int row = 0; row < image.rows; row++) {
		for (int column = 0; column < image.cols; column++) {
			result.at<uchar>(row, column) = RGBtoGrayscale(image.at<Vec3b>(row, column));
		}
	}
	return result;
}

void computeCorrespondingPixelCount(){
	for(int disparity = 0; disparity<MAX_DISPARITY; disparity++){
		correspondingPixelCount[disparity] = resizedImgLeft.rows * (resizedImgLeft.cols - disparity);
	}
}

Point computeBaseToMatch(Point base, int disparity){
	return Point(base.x, base.y - disparity);
}

void computeMutualProbabilityDistribution(){
	mutualProbabilityDistribution = allocate3D(MAX_DISPARITY, 256, 256);
	for(int disparity = 0; disparity < MAX_DISPARITY; disparity++){
		for(int row = 0; row < resizedImgLeft.rows; row++){
			for(int col = 0; col < resizedImgLeft.cols; col++){
				Point match = computeBaseToMatch(Point(row, col), disparity);
				if(isInside(match.x, match.y, resizedImgLeft.rows, resizedImgLeft.cols)){
					uchar leftIntensity = resizedImgLeft.at<uchar>(row, col);
					uchar rightIntensity = resizedImgRight.at<uchar>(match.x, match.y);
					mutualProbabilityDistribution[disparity][leftIntensity][rightIntensity]++;
				}
			}
		}
	}

	//normalized probability distribution
	for(int disparity = 0; disparity < MAX_DISPARITY; disparity++){
		for(int row = 0; row < 256; row++){
			for(int col = 0; col < 256; col++){
					mutualProbabilityDistribution[disparity][row][col] /= (correspondingPixelCount[disparity] * 1.0);
			}
		}
	}
}

void computeIndividualProbabilityLeft(){
	individualProbabilityLeft = allocate2D(MAX_DISPARITY, 256);

	for(int disparity = 0; disparity < MAX_DISPARITY; disparity++){
		for(int row = 0; row < 256; row++){
			for(int col = 0; col < 256; col++){
				individualProbabilityLeft[disparity][row] += mutualProbabilityDistribution[disparity][row][col];
			}
		}
	}
}

void computeIndividualProbabilityRight(){
	individualProbabilityRight = allocate2D(MAX_DISPARITY, 256);

	for(int disparity = 0; disparity < MAX_DISPARITY; disparity++){
		for(int row = 0; row < 256; row++){
			for(int col = 0; col < 256; col++){
				individualProbabilityRight[disparity][col] += mutualProbabilityDistribution[disparity][row][col];
			}
		}
	}
}

double** computeIndividulEntropy(double** inidvidualProbability){
	double** individualEntropy = allocate2D(MAX_DISPARITY, 256);
	double* intermediaryResult = (double*)calloc(256, sizeof(double));

	for(int disparity = 0; disparity < MAX_DISPARITY; disparity++){
		for(int intensity = 3; intensity < 253; intensity++){
			intermediaryResult[intensity] = 0;
			for(int i = -3; i <= 3; i++){
				intermediaryResult[intensity] += gaussian1D[i+3]*inidvidualProbability[disparity][intensity+i];
			}
		}
		for(int intensity = 0; intensity < 256; intensity++){
			if(intermediaryResult[intensity] > 0){
				intermediaryResult[intensity] = log(intermediaryResult[intensity] * 1/(-1.0 * correspondingPixelCount[disparity]));
			}
		}
		for(int intensity = 3; intensity < 253; intensity++){
			for(int i = -3; i <= 3; i++){
				individualEntropy[disparity][intensity] += gaussian1D[i+3]*intermediaryResult[intensity+i];
			}
		}
	}
	return individualEntropy;
}

void computeMutualEntropy(){
	mutualEntropy = allocate3D(MAX_DISPARITY, 256, 256);
	double** intermediaryResult = allocate2D(256, 256);

	for(int disparity = 0; disparity < MAX_DISPARITY; disparity++){

		for(int leftIntensity = 2; leftIntensity < 254; leftIntensity++){
			for(int rightIntensity = 2; rightIntensity < 254; rightIntensity++){
				intermediaryResult[leftIntensity][rightIntensity] = 0;
				for(int i = -2; i <= 2; i++){
					for(int j = -2; j <= 2; j++){
						intermediaryResult[leftIntensity][rightIntensity] += gaussian2D[i+2][j+2] * mutualProbabilityDistribution[disparity][leftIntensity+i][rightIntensity+j];
					}
				}
			}
		}

		for(int leftIntensity = 0; leftIntensity < 256; leftIntensity++){
			for(int rightIntensity = 0; rightIntensity < 256; rightIntensity++){
				if(intermediaryResult[leftIntensity][rightIntensity] != 0){
					intermediaryResult[leftIntensity][rightIntensity] = log(intermediaryResult[leftIntensity][rightIntensity] * 1/(-1.0 * correspondingPixelCount[disparity]));
				}
			}
		}

		for(int leftIntensity = 2; leftIntensity < 254; leftIntensity++){
			for(int rightIntensity = 2; rightIntensity < 254; rightIntensity++){
				for(int i = -2; i <= 2; i++){
					for(int j = -2; j <= 2; j++){
						mutualEntropy[disparity][leftIntensity][rightIntensity] += gaussian2D[i+2][j+2] * intermediaryResult[leftIntensity+i][rightIntensity+j];
					}
				}
			}
		}
	}
}

void computeMutualInformation(){
	mutualInformation = allocate3D(MAX_DISPARITY, 256, 256);
	for(int disparity = 0; disparity < MAX_DISPARITY; disparity++){
		for(int leftIntensity = 0; leftIntensity < 256; leftIntensity++){
			for(int rightIntensity = 0; rightIntensity < 256; rightIntensity++){
				mutualInformation[disparity][leftIntensity][rightIntensity] = mutualProbabilityDistribution[disparity][leftIntensity][rightIntensity] - individualEntropyLeft[disparity][leftIntensity] - individualEntropyRight[disparity][rightIntensity];
			}
		}
	}
}

void computePath(int disparity, int row, int col, int dx, int dy) {
	if (frequency[disparity][row][col] == 0) {
		Point match = computeBaseToMatch(Point(row, col), disparity);
		if (isInside(match.x, match.y, resizedImgLeft.rows, resizedImgLeft.cols)) {
			uchar leftIntensity = resizedImgLeft.at<uchar>(row, col);
			uchar rightIntensity = resizedImgRight.at<uchar>(match.x, match.y);
			
			double param1, param2, param3, param4, param5, param6;
			param1 = param2 = param3 = param4 = param5 = param6 = 1000000;

			param1 = costMatrix[disparity][row][col];//mutualEntropy[disparity][leftIntensity][rightIntensity];

			if (isInside(row - dx, col - dy, resizedImgLeft.rows, resizedImgLeft.cols)) {
				computePath(disparity, row - dx, col - dy, dx, dy);
				param2 = paths[disparity][row - dx][col - dy];
			
				if ((disparity - 1) >= 0) {
					computePath(disparity - 1, row - dx, col - dy, dx, dy);
					param3 = paths[disparity - 1][row - dx][col - dy];
				}
				
				if ((disparity + 1) < MAX_DISPARITY) {
					computePath(disparity + 1, row - dx, col - dy, dx, dy);
					param4 = paths[disparity + 1][row - dx][col - dy];
				}

				for (int disp = 0; disp < MAX_DISPARITY; disp++) {
					computePath(disp, row - dx, col - dy, dx, dy);
					if (paths[disp][row - dx][row - dy] < param5) {
						param5 = paths[disp][row - dx][row - dy];
					}
				}

				param3 += SMALL_PENALTY;
				param4 += SMALL_PENALTY;
				param5 += BIG_PENALTY;

				param6 = param5;
			}
			paths[disparity][row][col] = param1 + min(param2, min(param3, min(param4, param5))) - param6;
			
		}
	}
	frequency[disparity][row][col] = 1;
}

void semiglobalMatching() {
	int dx[] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	int dy[] = { 0, -1, -1, -1, 0, 1, 1, 1 };

	cost = allocate3D(MAX_DISPARITY, resizedImgLeft.rows, resizedImgLeft.cols);

	for (int i = 0; i < 8; i++) {
		paths = allocate3D(MAX_DISPARITY, resizedImgLeft.rows, resizedImgLeft.cols);
		frequency = allocate3D(MAX_DISPARITY, resizedImgLeft.rows, resizedImgLeft.cols);

		for (int disparity = 0; disparity < MAX_DISPARITY; disparity++) {
			for (int row = 0; row < resizedImgLeft.rows; row++) {
				for (int col = 0; col < resizedImgLeft.cols; col++) {
					if (frequency[disparity][row][col] == 0) {
						computePath(disparity, row, col, dx[i], dy[i]);
					}
				}
			}
		}

		for (int disparity = 0; disparity < MAX_DISPARITY; disparity++) {
			for (int row = 0; row < resizedImgLeft.rows; row++) {
				for (int col = 0; col < resizedImgLeft.cols; col++) {
					cost[disparity][row][col] += paths[disparity][row][col];
				}
			}
		}
		free3D(paths, MAX_DISPARITY, resizedImgLeft.rows);
		free3D(frequency, MAX_DISPARITY, resizedImgLeft.rows);
	}
}

Mat computeDisparityMap() {
	Mat disparityMap(resizedImgLeft.rows, resizedImgLeft.cols, CV_8UC1);

	for (int row = 0; row < resizedImgLeft.rows; row++) {
		for (int col = 0; col < resizedImgLeft.cols; col++) {
			double mini = 100000;
			int minDisparity = 0;
			for (int disparity = 0; disparity < MAX_DISPARITY; disparity++) {
				if (cost[disparity][row][col] < mini) {
					mini = cost[disparity][row][col];
					minDisparity = disparity;
				}
			}
			disparityMap.at<uchar>(row, col) = minDisparity / (1.0 * MAX_DISPARITY) * 255;
		}
	}
	return disparityMap;
}

double computeCostWindow(int disparity, int row, int col) {
	int count = 0;
	double sum = 0;

	for (int i = -3; i < 4; i++) {
		for (int j = -3; j < 4; j++) {
			Point matchPoint = computeBaseToMatch(Point(col, row), disparity);
			if (isInside(matchPoint.y, matchPoint.x, resizedImgLeft.rows,resizedImgRight.cols)){
				count++;
				float difference = resizedImgLeft.at<uchar>(row, col) - resizedImgRight.at<uchar>(matchPoint.y, matchPoint.x);
				sum += difference * difference;
			}
		}
	}
	sum /= (count * 1.0);
	return sum;
}

void computeCostWithWindow() {
	costMatrix = allocate3D(MAX_DISPARITY, resizedImgLeft.rows, resizedImgLeft.cols);

	for (int disparity = 0; disparity < MAX_DISPARITY; disparity++) {
		for (int row = 0; row < resizedImgLeft.rows; row++) {
			for (int col = 0; col < resizedImgLeft.cols; col++) {
				costMatrix[disparity][row][col] = computeCostWindow(disparity, row, col);
			}
		}
	}
}

Mat computeStereoMatching() {
	populateGaussian2D();
	computeCorrespondingPixelCount();

	computeMutualProbabilityDistribution();
	computeIndividualProbabilityLeft();
	computeIndividualProbabilityRight();

	individualEntropyLeft = computeIndividulEntropy(individualProbabilityLeft);
	individualEntropyRight = computeIndividulEntropy(individualProbabilityRight);
	computeMutualEntropy();

	computeMutualInformation();

	computeCostWithWindow();
	printf("Computed mutual information");
	semiglobalMatching();

	return computeDisparityMap();
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf("1 - Stereo-matching using Semi-global matching - choose left and right image.\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d", &op);
		switch (op) {
		case 1: {
				while (openFileDlg(fname)) {
					Mat leftImage;
					leftImage = imread(fname, CV_LOAD_IMAGE_COLOR);
					while (openFileDlg(fname)) {
						Mat rightImage;
						rightImage = imread(fname, CV_LOAD_IMAGE_COLOR);
						resizeImg(leftImage, resizedImgLeft, 600, true);
						resizeImg(rightImage, resizedImgRight, 600, true);
						Mat disparityMap = computeStereoMatching();
						imshow("DisparityMap", disparityMap);
						imwrite("result.png", disparityMap);
						waitKey(0);
					}
				}
			}
		}
	} while (op != 0);
	return 0;
}