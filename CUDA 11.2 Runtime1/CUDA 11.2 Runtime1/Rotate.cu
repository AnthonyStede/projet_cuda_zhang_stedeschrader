#include "cuda_runtime.h"

#include <windows.h>
#include <iostream>
#include <vector>
#include <cstring>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include <opencv2/cudaarithm.hpp>

using namespace cv;
using namespace std;
using namespace cv::cuda;

void rotateImage(const Mat &_src, Mat &_dst, int ind)
{
  for( int i = 0; i < _dst.rows; i++ )
    {
    for( int j = 0; j < _dst.cols; j++ )
    {
        switch( ind )
        {
        case 0:
            if( j > _dst.cols*0.25 && j < _dst.cols*0.75 && i > _dst.rows*0.25 && i < _dst.rows*0.75 )
            {
                _dst.at<float>(i, j) = 2*( j - _dst.cols*0.25f ) + 0.5f;
                _src.at<float>(i, j) = 2*( i - _dst.rows*0.25f ) + 0.5f;
            }
            else
            {
                _dst.at<float>(i, j) = 0;
                _src.at<float>(i, j) = 0;
            }
            break;
        case 1:
            _dst.at<float>(i, j) = (float)j;
            _src.at<float>(i, j) = (float)(_dst.rows - i);
            break;
        case 2:
            _dst.at<float>(i, j) = (float)(_dst.cols - j);
            _src.at<float>(i, j) = (float)i;
            break;
        case 3:
            _dst.at<float>(i, j) = (float)(_dst.cols - j);
            _src.at<float>(i, j) = (float)(_dst.rows - i);
            break;
        default:
            break;
        }
      }
    }
    ind = (ind+1) % 4;
}

bool initCUDA()
{
	int count;
	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;
	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}

	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);
	return true;
}


__global__ void kernel(uchar* _src_dev, uchar * _dst_dev, int _src_step, int _dst_step,
	int _src_rows, int _src_cols, int _dst_rows, int _dst_cols)
{
	auto i = blockIdx.x;
	auto j = blockIdx.y;

	double fRows = _dst_rows / (float)_src_rows;
	double fCols = _dst_cols / (float)_src_cols;

	int pX = 0;
	int pY = 0;

	pX = (int)(i / fRows);
	pY = (int)(j / fCols);
	if (pX < _src_rows && pX >= 0 && pY < _src_cols && pY >= 0) {
		*(_dst_dev + i * _dst_step + 3 * j + 0) = *(_src_dev + pX * _src_step + 3 * pY);
		*(_dst_dev + i * _dst_step + 3 * j + 1) = *(_src_dev + pX * _src_step + 3 * pY + 1);
		*(_dst_dev + i * _dst_step + 3 * j + 2) = *(_src_dev + pX * _src_step + 3 * pY + 2);

	}

}


void rotateImageGpu(const Mat &_src, Mat &_dst, int ind)
{
	_dst = Mat(s, CV_8UC3);
	uchar *src_data = _src.data;
	int width = _src.cols;
	int height = _src.rows;
	uchar *src_dev, *dst_dev;
  cv::cuda::GpuMat gpu_im ;
  gpu_im.upload( _src );
  cv::Size size = _src.size();
  cv::cuda::GpuMat gpu_im_rot ;
  cv::cuda::rotate( gpu_im, gpu_im_rot, cv::Size( size.height, size.width ), -90, size.height-1, 0, cv::INTER_LINEAR  );
  gpu_im_rot.download(_src);
  cv::imwrite( "out.png", _src );

}


int main()
{
	Mat src = cv::imread("1.jpg", 1);
	Mat dst_cpu;
  Mat map_x(src.size(), CV_32FC1);
  Mat map_y(src.size(), CV_32FC1);

	double start = GetTickCount();
  int ind = 0;
  for(;;)
  {
	 rotateImage(map_y, map_x, ind);
   remap( src, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0) );
   imshow( remap_window, dst );
   char c = (char)waitKey( 1000 );
   if( c == 27 )
    {
    break;
    }
  }
	double  end = GetTickCount();

	cout << "cpu cost time��" << end - start << "\n";

	initCUDA();

	Mat dst_gpu;

	start = GetTickCount();
	rotateImageGpu(src, dst_gpu, Size(src.cols * 2, src.rows * 2));
	end = GetTickCount();
	cout << "gpu cost time��" << end - start << "\n";

	cv::imshow("Demo", dst_cpu);
	waitKey(0);

	return 0;
}
