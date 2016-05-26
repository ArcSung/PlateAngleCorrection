// PlateAngleCorrection.h

#ifndef PlateAngleCorrection_H
#define PlateAngleCorrection_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv;

// global variables ///////////////////////////////////////////////////////////////////////////////
const cv::Size GAUSSIAN_SMOOTH_FILTER_SIZE = cv::Size(5, 5);
const int ADAPTIVE_THRESH_BLOCK_SIZE = 19;
const int ADAPTIVE_THRESH_WEIGHT = 9;

// function prototypes ////////////////////////////////////////////////////////////////////////////

void PlateAngleCorrection(cv::Mat &SrcImg, cv::Mat &Dstimg);

void preprocess(cv::Mat imgOriginal, cv::Mat &imgGrayscale, cv::Mat &imgThresh);

cv::Mat extractValue(cv::Mat imgOriginal);

cv::Mat maximizeContrast(cv::Mat imgGrayscale);

void rotateImage(const Mat &input, Mat &output, double alpha, double beta, double gamma, double dx, double dy, double dz, double f);

std::vector<Point2f> ReverserMat(std::vector<Point2f> input);

void FindPlateCorner(Mat image, Mat mgray, Mat bin_img);

#endif	// PlateAngleCorrection_H

