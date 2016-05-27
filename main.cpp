#include <opencv2/core/utility.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "PlateAngleCorrection.h"
#include "Main.h"


//PlateAngleCorrection *_PlateAngle;


// Main -------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	//set up matrices for storage
	Mat srcImage, dst_img;
    //_PlateAngle = new PlateAngleCorrection();

	if(argc == 1)
	{	
	   	srcImage = imread("Wilbur/IMG_2654.JPG");
	   	resize( srcImage, srcImage, Size(1280, 720) );
	   	imshow("Wilbur/IMG_2654.JPG", srcImage);
	   	std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(srcImage);
	   	//_PlateAngle->Correction(srcImage, dst_img);
	   	waitKey(0);
	}   

    for (int i = 2; i < argc; ++i)
    {
    	srcImage = imread(argv[i]);
    	resize( srcImage, srcImage, Size(1280, 720) );
    	imshow("argv[i]", srcImage);
    	std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(srcImage);
    	cv::destroyAllWindows();
    	//_PlateAngle->Correction(srcImage, dst_img);
    	//waitKey(0);
    }	


	return 0;
}