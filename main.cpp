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

using namespace std;
PlateAngleCorrection *_PlateAngle;


// Main -------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	//set up matrices for storage
	Mat srcImage, dst_img;
    _PlateAngle = new PlateAngleCorrection();

	if(argc == 1)
	{	
	   	srcImage = imread("1280x720/2016-03-29_16-14-50.jpg");
	   	resize( srcImage, srcImage, Size(1280, 720) );
	   	imshow("Wilbur/IMG_2654.JPG", srcImage);
	   	std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(srcImage);
	   	//_PlateAngle->Correction(srcImage, dst_img);
	   	waitKey(0);
	}   

    for (int i = 2; i < argc; ++i)
    {
    	srcImage = imread(argv[i]);
    	//resize( srcImage, srcImage, Size(1280, 720) );
    	cout << argv[i]<< endl;
    	imshow(argv[i], srcImage);
    	//std::vector<PossiblePlate> vectorOfPossiblePlates = detectPlatesInScene(srcImage);
    	_PlateAngle->Correction(srcImage, dst_img);
    	waitKey(0);
    	cv::destroyAllWindows();
    }	


	return 0;
}