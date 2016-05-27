// PlateAngleCorrection.h

#ifndef PlateAngleCorrection_H
#define PlateAngleCorrection_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv;

// global variables ///////////////////////////////////////////////////////////////////////////////

// function prototypes ////////////////////////////////////////////////////////////////////////////
class PlateAngleCorrection{

public:
	void Correction(cv::Mat &SrcImg, cv::Mat &Dstimg);

	void rotateImage(const Mat &input, Mat &output, double alpha, double beta, double gamma, double dx, double dy, double dz, double f);

private:
	void preprocess(cv::Mat imgOriginal, cv::Mat &imgGrayscale, cv::Mat &imgThresh);

	cv::Mat extractValue(cv::Mat imgOriginal);

	cv::Mat maximizeContrast(cv::Mat imgGrayscale);

	std::vector<Point2f> ReverserMat(std::vector<Point2f> input);

	void FindPlateCorner(Mat image, Mat mgray, Mat bin_img);

	Point2f L1, L2, L3, L4;
	
    int tlx, tly, brx, bry;

    cv::Size GAUSSIAN_SMOOTH_FILTER_SIZE = cv::Size(5, 5);
    int ADAPTIVE_THRESH_BLOCK_SIZE = 19;
    int ADAPTIVE_THRESH_WEIGHT = 9;

};

#endif	// PlateAngleCorrection_H

