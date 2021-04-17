#include"opencv2/opencv.hpp"
#include"opencv2/highgui/highgui.hpp"
#include"opencv2/imgproc/imgproc.hpp"
#include"cuda_runtime.h"
#include"iostream"
#include <math.h>
#include <device_launch_parameters.h>
#include <opencv2\imgproc\types_c.h>

using namespace std;
using namespace cv;
#define BLOCKDIM_X      32 
#define BLOCKDIM_Y      32  

#define GRIDDIM_X       256  

#define GRIDDIM_Y       256  
#define MASK_WIDTH      5  


__constant__ int templates[MASK_WIDTH*MASK_WIDTH]; // Allocate constant memory  

__global__ void GaussianFilter(uchar *d_in, uchar *d_out, int height, int width)
{
	int tidx = blockDim.x * blockIdx.x + threadIdx.x;
	int tidy = blockDim.y * blockIdx.y + threadIdx.y;

	int sum = 0;
	int index = 0;

	if (tidx > 2 && tidx < width - 2 && tidy>2 && tidy < height - 2) {
		for (int m = tidx - 2; m < tidx + 3; m++)
		{
			for (int n = tidy - 2; n < tidy + 3; n++)
			{
				sum += d_in[m*width + n] * templates[index++];
			}
		}
		if (sum / 273 < 0) {
			*(d_out + (tidx)*width + tidy) = 0;
		}
		else if (sum / 273 > 255) {
			*(d_out + (tidx)*width + tidy) = 255;
		}
		else {
			*(d_out + (tidx)*width + tidy) = sum / 273;
		}
	}
}

int main()
{

	Mat srcImg = imread("1.jpg");
	Mat src; 
	cvtColor(srcImg, src, CV_BGR2GRAY); // Convert to grayscale image

	imshow("src_image", src);

	uchar *d_in;
	uchar *d_out;

	int width = srcImg.rows;
	int height = srcImg.cols;

	int memsize = width*height*sizeof(uchar);

	cudaMalloc((void**)&d_in, width * height * sizeof(uchar));
	cudaMalloc((void**)&d_out, width * height * sizeof(uchar));

	cudaMemcpy(d_in, src.data, memsize, cudaMemcpyHostToDevice);// Data transfer from Host to Device

	int Gaussian[25] = { 1, 4, 7, 4, 1,
		4, 16, 26, 16, 4,
		7, 26, 41, 26, 7,
		4, 16, 26, 16, 4,
		1, 4, 7, 4, 1 };// The sum is 273
	cudaMemcpyToSymbol(templates, Gaussian, 25 * sizeof(int));

	int bx = int(ceil((double)width / BLOCKDIM_X)); // Distribution of grids and blocks
	int by = int(ceil((double)height / BLOCKDIM_Y));

	if (bx > GRIDDIM_X) bx = GRIDDIM_X;
	if (by > GRIDDIM_Y) by = GRIDDIM_Y;

	dim3 grid(bx, by);//   The structure of the grid
	dim3 block(BLOCKDIM_X, BLOCKDIM_Y);//The structure of the block

	//kernel--Gaussian filtering
	GaussianFilter <<< grid, block >>> (d_in, d_out, width, height);

	cudaMemcpy(src.data, d_out, memsize, cudaMemcpyDeviceToHost);// Data is sent back to the host

	imshow("cuda_gaussian", src);

	cudaFree(d_in);
	cudaFree(d_out);
	waitKey(0);
	return 0;
}