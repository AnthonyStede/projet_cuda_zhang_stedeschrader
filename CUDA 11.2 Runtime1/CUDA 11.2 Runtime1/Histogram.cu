#include "cuda_runtime.h"

#include <windows.h>
#include <iostream>
#include <vector>
#include <cstring>

#include <device_launch_parameters.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



using namespace cv;
using namespace std;

__global__ void Histogram_CUDA(unsigned char* Image, int* Histogram);

void Histogram_Calculation_CUDA(unsigned char* Image, int Height, int Width, int Channels, int* Histogram){
	unsigned char* Dev_Image = NULL;
	int* Dev_Histogram = NULL;

	cudaMalloc((void**)&Dev_Image, Height * Width * Channels);
	cudaMalloc((void**)&Dev_Histogram, 256 * sizeof(int));

	cudaMemcpy(Dev_Image, Image, Height * Width * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(Dev_Histogram, Histogram, 256 * sizeof(int), cudaMemcpyHostToDevice);

	dim3 Grid_Image(Width, Height);
	Histogram_CUDA << <Grid_Image, 1 >> >(Dev_Image, Dev_Histogram);

	cudaMemcpy(Histogram, Dev_Histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(Dev_Histogram);
	cudaFree(Dev_Image);
}

__global__ void Histogram_CUDA(unsigned char* Image, int* Histogram){
	int x = blockIdx.x;
	int y = blockIdx.y;

	int Image_Idx = x + y * gridDim.x;

	atomicAdd(&Histogram[Image[Image_Idx]], 1);
}


int main()
{
	Mat Input_Image = imread("1.jpg", 0);

	cout << "Image Height: " << Input_Image.rows << ", Image Width: " << Input_Image.cols << ", Image Channels: " << Input_Image.channels() << endl;

	int Histogram_GrayScale[256] = { 0 };

	Histogram_Calculation_CUDA(Input_Image.data, Input_Image.rows, Input_Image.cols, Input_Image.channels(), Histogram_GrayScale);

	imwrite("Histogram_Image.png", Input_Image);

	for (int i = 0; i < 256; i++){
		cout << "Histogram_GrayScale[" << i << "]: " << Histogram_GrayScale[i] << endl;
	}
	system("pause");
	return 0;
}
