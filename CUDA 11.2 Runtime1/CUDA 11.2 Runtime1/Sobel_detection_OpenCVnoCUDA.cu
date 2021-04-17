
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <opencv2\opencv.hpp>
#include <iostream>

#include <windows.h>   
#include <vector>
#include <cstring>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2\imgproc\types_c.h>

using namespace std;
using namespace cv;
 
// define sobel  kernel  
const int sobel_kernel_y[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
const int sobel_kernel_x[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };

// 3x3 position 
const int locations[9][2] = { { -1, -1 }, { -1, 0 }, { -1, 1 },
{ 0, -1 }, { 0, 0 }, { 0, 1 },
{ 1, -1 }, { 1, 0 }, { 1, 1 } };

// 8 neighbors
const int neighbors[8][2] = { { -1, -1 }, { -1, 0 }, { -1, 1 }, { 0, -1 }, { 0, 1 }, { 1, -1 }, { 1, 0 }, { 1, 1 } };


int main()
{

	Mat src = imread("E:\\1.jpg");
	Mat image = src.clone();
	GaussianBlur(src, image, Size(3, 3), 1.5);

	Mat gray;
	cvtColor(image, gray, CV_BGR2GRAY);


	Mat sobel_x = Mat::zeros(gray.rows, gray.cols, CV_8UC1);
	Mat sobel_y = Mat::zeros(gray.rows, gray.cols, CV_8UC1);
	Mat sobel_xy = Mat::zeros(gray.rows, gray.cols, CV_64FC1);

	int i, j, k;
	// compute sobel_x, sobel_y  
	for (i = 1; i < gray.rows - 1; i++)
		for (j = 1; j < gray.cols - 1; j++)
		{
			int temp_x = 0, temp_y = 0;
			for (k = 0; k < 9; k++)
			{
				temp_x += sobel_kernel_x[k] * gray.at<uchar>(i + locations[k][0], j + locations[k][1]);
				temp_y += sobel_kernel_y[k] * gray.at<uchar>(i + locations[k][0], j + locations[k][1]);
			}

			sobel_x.at<uchar>(i, j) = temp_x;
			sobel_y.at<uchar>(i, j) = temp_y;
			//sobel_xy.at<double>(i, j) = sqrt(temp_x * temp_x + temp_y * temp_y);
			sobel_xy.at<double>(i, j) = abs(temp_x) + abs(temp_y);
		}

	Mat directions = Mat::zeros(gray.rows, gray.cols, CV_64FC1);
	// compute direction  
	for (i = 1; i < gray.rows - 1; i++)
		for (j = 1; j < gray.cols - 1; j++)
		{
			// The value range of atan2 is [-pi,pi]
			float t = atan2(sobel_y.at<uchar>(i, j), sobel_x.at<uchar>(i, j));
			if (t < 0)
			{
				t += CV_PI;
			}
			directions.at<double>(i, j) = t;
		}

	float t = 0;
	// Non-maximum suppression
	for (i = 1; i < gray.rows - 1; i++)
		for (j = 1; j < gray.cols - 1; j++)
		{

			t = directions.at<double>(i, j);
			// 0 - 22.5,   
			if (((t >= 0) && (t < CV_PI / 8.0)) || ((t >= 7.0 * CV_PI / 8.0) && (t < CV_PI)))
			{
				if ((sobel_xy.at<double>(i, j) < sobel_xy.at<double>(i, j + 1)) ||
					(sobel_xy.at<double>(i, j) < sobel_xy.at<double>(i, j - 1)))
				{
					sobel_xy.at<double>(i, j) = 0;
				}
			}
			// 22.5 - 67.5  
			else if ((t >= CV_PI / 8.0) && (t < 3.0 * CV_PI / 8.0))
			{
				if ((sobel_xy.at<double>(i, j) < sobel_xy.at<double>(i - 1, j + 1)) ||
					(sobel_xy.at<double>(i, j) < sobel_xy.at<double>(i + 1, j - 1)))
				{
					sobel_xy.at<double>(i, j) = 0;
				}
			}
			// 67.5 - 112.5  
			else if ((t >= 3.0 * CV_PI / 8.0) && (t < 5.0 * CV_PI / 8.0))
			{
				if ((sobel_xy.at<double>(i, j) < sobel_xy.at<double>(i - 1, j)) ||
					(sobel_xy.at<double>(i, j) < sobel_xy.at<double>(i + 1, j)))
				{
					sobel_xy.at<double>(i, j) = 0;
				}
			}
			// 112.5 - 157.5  
			else if ((t >= 5.0 * CV_PI / 8.0) && (t < 7.0 * CV_PI / 8.0))
			{
				if ((sobel_xy.at<double>(i, j) < sobel_xy.at<double>(i - 1, j - 1)) ||
					(sobel_xy.at<double>(i, j) < sobel_xy.at<double>(i + 1, j + 1)))
				{
					sobel_xy.at<double>(i, j) = 0;
				}
			}
		}

	// Dual threshold filtering  
	float lower_t = 30;
	float upper_t = 100;

	Mat My_canny = Mat::zeros(sobel_xy.rows, sobel_xy.cols, CV_8UC1);
	Mat sobel_xy_mask = Mat::zeros(sobel_xy.rows, sobel_xy.cols, CV_32FC1);
	sobel_xy.copyTo(sobel_xy_mask);

	// Judge what can be judged by the two thresholds
	for (i = 1; i < sobel_xy_mask.rows - 1; i++)
		for (j = 1; j < sobel_xy_mask.cols - 1; j++)
		{
			if (sobel_xy_mask.at<double>(i, j) > upper_t)
			{
				My_canny.at<uchar>(i, j) = 255;
				sobel_xy_mask.at <double>(i, j) = 0;   // Set to 0 when it is used up, which is convenient for later detection
			}
			if (sobel_xy_mask.at<double>(i, j) < lower_t)
			{
				sobel_xy.at<double>(i, j) = 0;
			}
		}

	// Enhance the part in the middle of the two thresholds
	for (i = 1; i < sobel_xy_mask.rows - 1; i++)
		for (j = 1; j < sobel_xy_mask.cols - 1; j++)
		{

			if (sobel_xy_mask.at<double>(i, j) >= lower_t)
			{
				// Traverse eight neighbor to see if there are 255 in the eight neighborhoods
				for (k = 0; k < 8; k++)
				{
					if (My_canny.at<uchar>(i + neighbors[k][0], j + neighbors[k][1]) == 255)
					{
						My_canny.at<uchar>(i, j) = 255;
						sobel_xy_mask.at<double>(i, j) = 0;
						break;
					}
				}

			}

		}


	Mat canny_sys;
	Canny(gray, canny_sys, 30, 100);

	namedWindow("gray", 0);
	resizeWindow("sobel_x", 300, 300);
	namedWindow("sobel_x", 0);
	resizeWindow("sobel_y", 300, 300);
	namedWindow("sobel_y", 0);
	resizeWindow("sobel_xy", 300, 300);
	namedWindow("sobel_xy", 0);
	resizeWindow("sobel_xy", 300, 300);
	namedWindow("My_canny", 0);
	resizeWindow("My_canny", 300, 300);
	namedWindow("canny_sys", 0);
	resizeWindow("canny_sys", 300, 300);


	imshow("gray", gray);
	imshow("sobel_x", sobel_x);
	imshow("sobel_y", sobel_y);
	imshow("sobel_xy", sobel_xy);
	imshow("My_canny", My_canny);
	imshow("canny_sys", canny_sys); // show all images

	waitKey();

	return 0;
}


