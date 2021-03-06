// PlateAngleCorrection.cpp
#include "PlateAngleCorrection.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

#define Debug_show true


bool PlateAngleCorrection::Correction(cv::Mat &SrcImg, cv::Mat &Dstimg) {
  Mat gray_img, bw_img, thr_img, edge_img;
  Mat sobelX, sobelY; 

  preprocess(SrcImg, gray_img, bw_img);
  Mat element(3,3,CV_8U,Scalar(255));  
  Sobel(gray_img,sobelX,CV_8U,1,0);
  Sobel(gray_img,sobelY,CV_8U,0,1);

  edge_img= abs(sobelX)+abs(sobelY);  

  dilate(edge_img,edge_img,element);  
  erode(edge_img,edge_img,element);

  gray_img = gray_img - edge_img ;
  line(gray_img, Point(0, 0), Point(gray_img.cols- 1, 0), CV_RGB(0,0,0), 2 );
  line(gray_img, Point(0, 0), Point(0, gray_img.rows- 1), CV_RGB(0,0,0), 2 );
  line(gray_img, Point(gray_img.cols- 1, 0), Point(gray_img.cols- 1, gray_img.rows- 1), CV_RGB(0,0,0), 2 );
  line(gray_img, Point(0, gray_img.rows - 1), Point(gray_img.cols- 1, gray_img.rows- 1), CV_RGB(0,0,0), 2 );

  threshold(gray_img, thr_img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

  if(!FindPlateCorner(SrcImg, gray_img, thr_img))
    return false;

  int ROIW = L2.x - L1.x - 1;
  int ROIH = L4.y - L1.y - 1;

  if(ROIW <= 0 || ROIH <= 0)
    return false;

  Mat SrcROI = SrcImg(Rect(tlx, tly, ROIW, ROIH));

  //設定變換[之前]與[之後]的坐標 (左上,左下,右下,右上)
  cv::Point2f pts1[] = {L1,L4,L3,L2};
  cv::Point2f pts2[] = {cv::Point2f(0,0),cv::Point2f(0,ROIH),cv::Point2f(ROIW,ROIH),cv::Point2f(ROIW,0)};

  cv::Mat perspective_matrix = cv::getPerspectiveTransform(pts1, pts2);
                
  // 變換
  cv::warpPerspective(SrcImg, Dstimg, perspective_matrix, Size(ROIW, ROIH), cv::INTER_LINEAR);

#if Debug_show  
  imshow("OpenCV edge_img", edge_img);
  imshow("OpenCV thr_img", thr_img);
  imshow("OpenCV gray_img", gray_img);
  imshow("OpenCV Dstimg", Dstimg);
  //waitKey(0);
#endif  

  return true;
}




///////////////////////////////////////////////////////////////////////////////////////////////////
void PlateAngleCorrection::preprocess(cv::Mat imgOriginal, cv::Mat &imgGrayscale, cv::Mat &imgThresh) {
	imgGrayscale = extractValue(imgOriginal);

	cv::Mat imgMaxContrastGrayscale = maximizeContrast(imgGrayscale);

	cv::Mat imgBlurred;

	cv::GaussianBlur(imgMaxContrastGrayscale, imgBlurred, GAUSSIAN_SMOOTH_FILTER_SIZE, 0);

	cv::adaptiveThreshold(imgBlurred, imgThresh, 255.0, CV_ADAPTIVE_THRESH_GAUSSIAN_C, 
    CV_THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
cv::Mat PlateAngleCorrection::extractValue(cv::Mat imgOriginal) {
	cv::Mat imgHSV;
	std::vector<cv::Mat> vectorOfHSVImages;
	cv::Mat imgValue;

	cv::cvtColor(imgOriginal, imgHSV, CV_BGR2HSV);

	cv::split(imgHSV, vectorOfHSVImages);

	imgValue = vectorOfHSVImages[2];

	return(imgValue);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
cv::Mat PlateAngleCorrection::maximizeContrast(cv::Mat imgGrayscale) {
	cv::Mat imgTopHat;
	cv::Mat imgBlackHat;
	cv::Mat imgGrayscalePlusTopHat;
	cv::Mat imgGrayscalePlusTopHatMinusBlackHat;

	cv::Mat structuringElement = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(3, 3));

	cv::morphologyEx(imgGrayscale, imgTopHat, CV_MOP_TOPHAT, structuringElement);
	cv::morphologyEx(imgGrayscale, imgBlackHat, CV_MOP_BLACKHAT, structuringElement);

	imgGrayscalePlusTopHat = imgGrayscale + imgTopHat;
	imgGrayscalePlusTopHatMinusBlackHat = imgGrayscalePlusTopHat - imgBlackHat;

	return(imgGrayscalePlusTopHatMinusBlackHat);
}

void PlateAngleCorrection::rotateImage(const Mat &input, Mat &output, double alpha, double beta, double gamma, double dx, double dy, double dz, double f)
  {
    alpha = (alpha - 90.)*CV_PI/180.;
    beta = (beta - 90.)*CV_PI/180.;
    gamma = (gamma - 90.)*CV_PI/180.;
    // get width and height for ease of use in matrices
    double w = (double)input.cols;
    double h = (double)input.rows;
    // Projection 2D -> 3D matrix
    Mat A1 = (Mat_<double>(4,3) <<
              1, 0, -w/2,
              0, 1, -h/2,
              0, 0,    0,
              0, 0,    1);
    // Rotation matrices around the X, Y, and Z axis
    Mat RX = (Mat_<double>(4, 4) <<
              1,          0,           0, 0,
              0, cos(alpha), -sin(alpha), 0,
              0, sin(alpha),  cos(alpha), 0,
              0,          0,           0, 1);
    Mat RY = (Mat_<double>(4, 4) <<
              cos(beta), 0, -sin(beta), 0,
              0, 1,          0, 0,
              sin(beta), 0,  cos(beta), 0,
              0, 0,          0, 1);
    Mat RZ = (Mat_<double>(4, 4) <<
              cos(gamma), -sin(gamma), 0, 0,
              sin(gamma),  cos(gamma), 0, 0,
              0,          0,           1, 0,
              0,          0,           0, 1);
    // Composed rotation matrix with (RX, RY, RZ)
    Mat R = RX * RY * RZ;
    // Translation matrix
    Mat T = (Mat_<double>(4, 4) <<
             1, 0, 0, dx,
             0, 1, 0, dy,
             0, 0, 1, dz,
             0, 0, 0, 1);
    // 3D -> 2D matrix
    Mat A2 = (Mat_<double>(3,4) <<
              f, 0, w/2, 0,
              0, f, h/2, 0,
              0, 0,   1, 0);
    // Final transformation matrix
    Mat trans = A2 * (T * (R * A1));
    // Apply matrix transformation
    warpPerspective(input, output, trans, input.size(), INTER_LANCZOS4);
  }

  std::vector<Point2f> ReverserMat(std::vector<Point2f> input) {

    std::vector<Point2f> tempMat;
     
    for( int i = 0; i < input.size(); i++ )  
    {  
      tempMat.push_back(input[input.size()-i-1]);  
    }  

    return(tempMat);
  }

  bool PlateAngleCorrection::FindPlateCorner(Mat image, Mat mgray, Mat bin_img){
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(bin_img,
    contours,
    hierarchy,
    cv::RETR_TREE,
    cv::CHAIN_APPROX_SIMPLE);

    Mat drawing = Mat::zeros( bin_img.size(), CV_8UC3 );

    int MaxSize = 0, MaxSizeId = 0;
    vector< vector< Point> >::iterator itc = contours.begin();

    for(;itc!=contours.end();){  //remove some contours size <= 100
      if(itc->size() <= 100)
        itc = contours.erase(itc);
      else
        ++itc;
    }

    if(contours.size() == 0)
      return false;

    //Find the max contour
    for( int i = 0; i< contours.size(); i++ ) // iterate through each contour. 
    {
      double a=contourArea( contours[i],false);  //  Find the area of contour
      if(a>MaxSize){
        MaxSize=a;
        MaxSizeId=i;                //Store the index of largest contour
      }
    } 

    vector< vector< Point> > contours_poly(contours.size());
    approxPolyDP( Mat(contours[MaxSizeId]), contours_poly[MaxSizeId], 3, true ); // let contours more smooth
  
    //find 4 conner in contour
    int minsum = 10000, maxsum = 0, mindiff = 10000, maxdiff = 0;

    for(int j=0;j<contours_poly[MaxSizeId].size();j++){
      if(contours_poly[MaxSizeId][j].x+ contours_poly[MaxSizeId][j].y < minsum)
      {
        //The left top point, x + y is minimum
        minsum = contours_poly[MaxSizeId][j].x+ contours_poly[MaxSizeId][j].y;
        L1 = Point2f(contours_poly[MaxSizeId][j].x, contours_poly[MaxSizeId][j].y);
      }
      if(contours_poly[MaxSizeId][j].x+ contours_poly[MaxSizeId][j].y > maxsum)
      {
        //The right bottom point, x + y is maxmum
        maxsum = contours_poly[MaxSizeId][j].x+ contours_poly[MaxSizeId][j].y;
        L3 = Point2f(contours_poly[MaxSizeId][j].x, contours_poly[MaxSizeId][j].y);
      }
      if(contours_poly[MaxSizeId][j].x - contours_poly[MaxSizeId][j].y < mindiff)
      {
        //The left bottom point, x - y is minmum
        mindiff = contours_poly[MaxSizeId][j].x - contours_poly[MaxSizeId][j].y;
        L4 = Point2f(contours_poly[MaxSizeId][j].x, contours_poly[MaxSizeId][j].y);
      }
      if(contours_poly[MaxSizeId][j].x - contours_poly[MaxSizeId][j].y > maxdiff)
      {
        //The right up point, x - y is maxmum
        maxdiff = contours_poly[MaxSizeId][j].x - contours_poly[MaxSizeId][j].y;
        L2 = Point2f(contours_poly[MaxSizeId][j].x, contours_poly[MaxSizeId][j].y);
      }
    }
  

#if Debug_show 
    drawContours(drawing, contours_poly, -1, Scalar(255.0, 255.0, 255.0));
    circle(drawing, L1, 3, CV_RGB(0, 0, 255), 3, CV_AA);
    circle(drawing, L2, 3, CV_RGB(0, 0, 255), 3, CV_AA);
    circle(drawing, L3, 3, CV_RGB(0, 0, 255), 3, CV_AA);
    circle(drawing, L4, 3, CV_RGB(0, 0, 255), 3, CV_AA);    
    imshow("drawing", drawing);
#endif    

    return true;
}



