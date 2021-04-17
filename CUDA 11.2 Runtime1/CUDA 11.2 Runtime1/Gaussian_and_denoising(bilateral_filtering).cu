#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include<cmath>
using namespace std;

#define MASK_WIDTH		5
__constant__ int d_const_Gaussian[MASK_WIDTH * MASK_WIDTH]; //Allocate constant memory

//Gaussian filtering
__global__ void GaussianFiltInCuda(unsigned char* dataIn, unsigned char* dataOut, cv::Size erodeElement, int imgWidth, int imgHeight)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x; 
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y; 

	int Index = xIndex + yIndex * imgWidth;

	int elementWidth = erodeElement.width;
	int elementHeight = erodeElement.height;
	int halfEW = elementWidth / 2;
	int halfEH = elementHeight / 2;

	//Initialization output
	dataOut[Index] = dataIn[Index];

	//Prevent cross-border  halfEW < xIndex < imgWidth-halfEW         halfEH < yIndex < imgHeight-halfEH
	if (xIndex > halfEW && xIndex < imgWidth - halfEW && yIndex > halfEH && yIndex < imgHeight - halfEH)
	{
		int sum = 0;
		for (int i = -halfEH; i < halfEH + 1; i++)
		{
			for (int j = -halfEW; j < halfEW + 1; j++)
			{
			
			   /* if (dataIn[(i + yIndex) * imgWidth + xIndex + j] < dataOut[yIndex * imgWidth + xIndex])
				{
					dataOut[yIndex * imgWidth + xIndex] = dataIn[(i + yIndex) * imgWidth + xIndex + j];
				}*/

				sum += dataIn[(i + yIndex) * imgWidth + xIndex + j] * d_const_Gaussian[(i + 2) * 5 + j + 2];

			}
		}

		if (sum / 273 < 0)
			dataOut[yIndex * imgWidth + xIndex] = 0;
		else if (sum / 273 > 255)
			dataOut[yIndex * imgWidth + xIndex] = 255;
		else
			dataOut[yIndex * imgWidth + xIndex] = sum / 273;

	}

}

//denoising (bilateral filtering)
__global__ void bilateralInCuda(unsigned char* dataIn, unsigned char* dataOut, cv::Size dilateElement, int imgWidth, int imgHeight)
{
	int xIndex = threadIdx.x + blockIdx.x * blockDim.x; 
	int yIndex = threadIdx.y + blockIdx.y * blockDim.y; 

	int elementWidth = dilateElement.width;   
	int elementHeight = dilateElement.height; 
	int halfEW = elementWidth / 2;
	int halfEH = elementHeight / 2;

	//Initialization output
	dataOut[yIndex * imgWidth + xIndex] = dataIn[yIndex * imgWidth + xIndex];

	//Prevent cross-border
	if (xIndex > halfEW && xIndex < imgWidth - halfEW && yIndex > halfEH && yIndex < imgHeight - halfEH)
	{
		int sum = 0;
		double num = 0;
		int sigm = 50;
		for (int i = -halfEH; i < halfEH + 1; i++)
		{
			for (int j = -halfEW; j < halfEW + 1; j++)
			{
				/*if (dataIn[(i + yIndex) * imgWidth + xIndex + j] > dataOut[yIndex * imgWidth + xIndex])
				{
					dataOut[yIndex * imgWidth + xIndex] = dataIn[(i + yIndex) * imgWidth + xIndex + j];
				}*/

				num = exp(-(double)((dataIn[(i + yIndex) * imgWidth + xIndex + j] - dataIn[yIndex * imgWidth + xIndex])* (dataIn[(i + yIndex) * imgWidth + xIndex + j] - dataIn[yIndex * imgWidth + xIndex]) / sigm / sigm) / 2);
				sum += (int)dataIn[(i + yIndex) * imgWidth + xIndex + j] * d_const_Gaussian[(i + 2) * 5 + j + 2] * num;
			}
		}

		if (sum / 273 < 0)
			dataOut[yIndex * imgWidth + xIndex] = 0;
		else if (sum / 273 > 255)
			dataOut[yIndex * imgWidth + xIndex] = 255;
		else
			dataOut[yIndex * imgWidth + xIndex] = sum / 273;
	}
}


int main()
{

	int dev = 0;
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, dev);
	std::cout << "Use GPU device " << dev << ": " << devProp.name << std::endl;
	std::cout << "Number of SM£º" << devProp.multiProcessorCount << std::endl;
	std::cout << "The shared memory size of each thread block£º" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
	std::cout << "Maximum number of threads per thread block£º" << devProp.maxThreadsPerBlock << std::endl;
	std::cout << "Maximum number of threads per EM£º" << devProp.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "Maximum number of warps per EM£º" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;

	cv::Mat grayImg = cv::imread("1.jpg", 0); 

	unsigned char* d_in;   
	unsigned char* d_out1; 
	unsigned char* d_out2; 

	int imgWidth = grayImg.cols;
	int imgHeight = grayImg.rows;

	cv::Mat GAUImg(imgHeight, imgWidth, CV_8UC1, cv::Scalar(0));  //Define empty image to store Gaussian filtering results
	cv::Mat BILAImg(imgHeight, imgWidth, CV_8UC1, cv::Scalar(0)); //Define an empty image to store the results of bilateral filtering

	//Allocate GPU memory for GPU variable pointers
	cudaMalloc((void**)&d_in, imgWidth * imgHeight * sizeof(unsigned char));
	cudaMalloc((void**)&d_out1, imgWidth * imgHeight * sizeof(unsigned char));
	cudaMalloc((void**)&d_out2, imgWidth * imgHeight * sizeof(unsigned char));

	//Copy CPU image data to GPU memory pointer variable
	cudaMemcpy(d_in, grayImg.data, imgWidth * imgHeight * sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(32, 32); //Define 32*32 dimension block thread block to improve the calculation speed as much as possible
	dim3 blocksPerGrid((imgWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(imgHeight + threadsPerBlock.y - 1) / threadsPerBlock.y); 


	cv::Size Element(5, 5);//Operator size

	int Gaussian[25] = { 1,4,7,4,1,
							4,16,26,16,4,
							7,26,41,26,7,
							4,16,26,16,4,
							1,4,7,4,1 };//sum is 273
	cudaMemcpyToSymbol(d_const_Gaussian, Gaussian, 25 * sizeof(int));

	//cuda Gaussian filter
	GaussianFiltInCuda << <blocksPerGrid, threadsPerBlock >> > (d_in, d_out1, Element, imgWidth, imgHeight);
	//cuda bilateral filtering
	bilateralInCuda << <blocksPerGrid, threadsPerBlock >> > (d_in, d_out2, Element, imgWidth, imgHeight);

	//Assign the GPU calculation result variable back to the host CPU(Device to host)
	cudaMemcpy(GAUImg.data, d_out1, imgWidth * imgHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	cudaMemcpy(BILAImg.data, d_out2, imgWidth * imgHeight * sizeof(unsigned char), cudaMemcpyDeviceToHost);


	cv::imshow("orgin", grayImg);
	
	cv::imshow("Gaussian", GAUImg);
	
	cv::imshow("bilateralFilter", BILAImg);
	cv::waitKey(100000);

	//Free
	cudaFree(d_in);
	cudaFree(d_out1);
	cudaFree(d_out2);

	return 0;
}