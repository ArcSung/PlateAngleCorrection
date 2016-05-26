#include <opencv2/core/utility.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "PlateAngleCorrection.h"





// Main -------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	//set up matrices for storage
	Mat srcImage, dst_img;
	if(argc > 1)
	   srcImage = imread(argv[1]);
	else
	   srcImage = imread("car_test2.jpg");
	PlateAngleCorrection(srcImage, dst_img);


	return 0;
}