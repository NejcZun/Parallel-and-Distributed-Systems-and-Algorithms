#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "timing.h"
#include <math.h>
#include "image.h"

#define BINS 256
#define MAX_SOURCE_SIZE	16384
#define image_file "slike/big.png"

// CPU --------------------------------------------------------------------------------------------------------
void printHistogram(unsigned long* histogram) {
	printf("Barva\tPojavitve\n");
	for (int i = 0; i < BINS; i++) {
		printf("%d\t%d\n", i, histogram[i]);
	}
}
void printHistogramValues(unsigned long* histogram) {
	for (int i = 0; i < BINS; i++) {
		printf("%d,", histogram[i]);
	}
}


void HistogramCPU(unsigned char* imageIn, unsigned long* histogram, int width, int height) {
	memset(histogram, 0, BINS * sizeof(unsigned long));
	//za vsak piksel v sliki
	for (int i = 0; i < (height); i++)
		for (int j = 0; j < (width); j++) {
			histogram[imageIn[i * width + j]]++;
		}
}


void CalculateCDF(unsigned long* histogram, unsigned long* cdf) {
	memset(cdf, 0, BINS * sizeof(unsigned int));
	// calculate cdf from histogram
	cdf[0] = histogram[0];
	for (int i = 1; i < BINS; i++) {
		cdf[i] = cdf[i - 1] + histogram[i];
	}
}

unsigned long findMin(unsigned long* cdf, unsigned long* histogram) {
	unsigned long min = 0;
	for (int i = 0; min == 0 && i < BINS; i++) {
		min = histogram[i];
	}
	return min;
}
unsigned char scale(unsigned long cdf, unsigned long cdfmin, unsigned long imageSize) {
	float scale;
	scale = (float)(cdf - cdfmin) / (float)(imageSize - cdfmin);
	scale = round(scale * (float)(BINS - 1));
	return (int)scale;
}

void Equalize(unsigned char* image, int width, int heigth, unsigned long* cdf, unsigned long* histogram) {
	unsigned long imageSize = width * heigth;
	unsigned long cdfmin = findMin(cdf, histogram);
	// Equalize
	for (int i = 0; i < heigth; i++) {
		for (int j = 0; j < width; j++) {
			//printf("[%d, %d]: %d\n", i, j, cdf[image[i * width + j]]);
			image[i * width + j] = scale(cdf[image[i * width + j]], cdfmin, imageSize);
			//printf("[%d, %d]: %d\n", i, j, image[i * width + j]);
		}
	}
}
unsigned char* CPU(void) {
	FIBITMAP* imageBitmap = FreeImage_Load(FIF_PNG, image_file, 0);
	FIBITMAP* imageBitmapGrey = FreeImage_ConvertToGreyscale(imageBitmap);
	int width = FreeImage_GetWidth(imageBitmapGrey);
	int height = FreeImage_GetHeight(imageBitmapGrey);

	// Allocate memory for raw image data , histogram , and CDF
	unsigned char* imageRaw = (unsigned char*)malloc(height * width * sizeof(unsigned long));
	unsigned long* histogram = (unsigned long*)malloc(BINS * sizeof(unsigned long));
	unsigned long* CDF = (unsigned long*)malloc(BINS * sizeof(unsigned long));

	// Convert image to raw pixel data
	FreeImage_ConvertToRawBits(imageRaw, imageBitmapGrey, width, 8, 0xFF, 0xFF, 0xFF, TRUE);

	// Histogram equalization steps :

	// 1. Create the histogram for the input grayscale image .
	HistogramCPU(imageRaw, histogram, width, height);
	// 2. Calculate the cumulative distribution histogram .
	CalculateCDF(histogram, CDF);
	// 3. Calculate the new gray - level values through the general histogram equalization formula and assign new pixel values
	Equalize(imageRaw, width, height, CDF, histogram);
	// Convert raw pixel data back to Freeimage format
	FIBITMAP* imageOutBitmap = FreeImage_ConvertFromRawBits(imageRaw, width, height, width, 8, 0xFF, 0xFF, 0xFF, TRUE);

	// Save image to file
	FreeImage_Save(FIF_PNG, imageOutBitmap, "CPU_output.png", 0);

	// Free memory
	FreeImage_Unload(imageBitmapGrey);
	FreeImage_Unload(imageBitmap);
	FreeImage_Unload(imageOutBitmap);

	return imageRaw;
}
// CPU --------------------------------------------------------------------------------------------------------