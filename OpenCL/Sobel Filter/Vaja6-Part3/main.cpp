#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>
#include "image.h"
#include "timing.h"

#define MAX_SOURCE_SIZE	16384
#define MAX(a,b) (((a)>(b))?(a):(b))

inline int getPixel(unsigned char* image, int width, int height, int y, int x){
	if (x < 0 || x >= width)
		return 0;
	if (y < 0 || y >= height)
		return 0;
	return image[y * width + x];
}

void sobelCPU(unsigned char* imageIn, unsigned char* imageOut, int width, int height){
	int i, j;
	int Gx, Gy;
	int tempPixel;

	//za vsak piksel v sliki
	for (i = 0; i < (height); i++)
		for (j = 0; j < (width); j++)
		{
			Gx = -getPixel(imageIn, width, height, i - 1, j - 1) - 2 * getPixel(imageIn, width, height, i - 1, j) -
				getPixel(imageIn, width, height, i - 1, j + 1) + getPixel(imageIn, width, height, i + 1, j - 1) +
				2 * getPixel(imageIn, width, height, i + 1, j) + getPixel(imageIn, width, height, i + 1, j + 1);
			Gy = -getPixel(imageIn, width, height, i - 1, j - 1) - 2 * getPixel(imageIn, width, height, i, j - 1) -
				getPixel(imageIn, width, height, i + 1, j - 1) + getPixel(imageIn, width, height, i - 1, j + 1) +
				2 * getPixel(imageIn, width, height, i, j + 1) + getPixel(imageIn, width, height, i + 1, j + 1);
			tempPixel = sqrt((float)(Gx * Gx + Gy * Gy));
			if (tempPixel > 255)
				imageOut[i * width + j] = 255;
			else
				imageOut[i * width + j] = tempPixel;
		}
}
void CPU() {
	FIBITMAP* imageBitmap = FreeImage_Load(FIF_JPEG, "slika.jpeg", 0);
	FIBITMAP* imageBitmapGrey = FreeImage_ConvertToGreyscale(imageBitmap);

	int width = FreeImage_GetWidth(imageBitmapGrey);
	int height = FreeImage_GetHeight(imageBitmapGrey);
	int pitch = FreeImage_GetPitch(imageBitmapGrey);


	unsigned char* imageIn = (unsigned char*)malloc(height * width * sizeof(unsigned char));
	unsigned char* imageOut = (unsigned char*)malloc(height * width * sizeof(unsigned char));
	FreeImage_ConvertToRawBits(imageIn, imageBitmapGrey, pitch, 8, 0xFF, 0xFF, 0xFF, TRUE);

	FreeImage_Unload(imageBitmapGrey);
	FreeImage_Unload(imageBitmap);
	BEGCLOCK(CPU)
	sobelCPU(imageIn, imageOut, width, height);
	ENDCLOCK(CPU)
	FIBITMAP* imageOutBitmap = FreeImage_ConvertFromRawBits(imageOut, width, height, pitch, 8, 0xFF, 0xFF, 0xFF, TRUE);
	FreeImage_Save(FIF_PNG, imageOutBitmap, "sobel_slika_CPU.png", 0);
	FreeImage_Unload(imageOutBitmap);
	free(imageOut);
	free(imageIn);
}
void GPU() {
	FIBITMAP* imageBitmap = FreeImage_Load(FIF_PNG, "FRI.png", 0);
	FIBITMAP* imageBitmapGrey = FreeImage_ConvertToGreyscale(imageBitmap);
	char ch;
	cl_int ret;
	int width = FreeImage_GetWidth(imageBitmapGrey);
	int height = FreeImage_GetHeight(imageBitmapGrey);
	int pitch = FreeImage_GetPitch(imageBitmapGrey);
	int size = MAX(width, height);
	printf("Width: %d, Height: %d", width, height);

	// Branje datoteke
	FILE* fp;
	char* source_str;
	size_t source_size;

	fp = fopen("kernel.cl", "r");
	if (!fp)
	{
		fprintf(stderr, ":-(#\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	source_str[source_size] = '\0';
	fclose(fp);


	unsigned char* imageIn = (unsigned char*)malloc(height * width * sizeof(unsigned char));
	unsigned char* imageOut = (unsigned char*)malloc(height * width * sizeof(unsigned char));

	FreeImage_ConvertToRawBits(imageIn, imageBitmapGrey, pitch, 8, 0xFF, 0xFF, 0xFF, TRUE);
	FreeImage_Unload(imageBitmapGrey);
	FreeImage_Unload(imageBitmap);

	// Podatki o platformi
	cl_platform_id platform_id[10];
	cl_uint	ret_num_platforms;
	char* buf;
	size_t	buf_len;
	ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);
	// max. "stevilo platform, kazalec na platforme, dejansko "stevilo platform

	// Podatki o napravi
	cl_device_id	device_id[10];
	cl_uint			ret_num_devices;
	// Delali bomo s platform_id[0] na GPU
	ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10, device_id, &ret_num_devices);
	// izbrana platforma, tip naprave, koliko naprav nas zanima
	// kazalec na naprave, dejansko "stevilo naprav

	// Kontekst
	cl_context context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &ret);
	// kontekst: vklju"cene platforme - NULL je privzeta, "stevilo naprav, 
	// kazalci na naprave, kazalec na call-back funkcijo v primeru napake
	// dodatni parametri funkcije, "stevilka napake

	// Ukazna vrsta
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);
	// kontekst, naprava, INORDER/OUTOFORDER, napake

	// Delitev dela
	size_t local_item_size[2] = { 16, 16 };
    size_t num_groups[2] = { (width-1)/ local_item_size[0] + 1, (height-1) / local_item_size[1]+1 };
    size_t global_item_size[2] = { num_groups[0] * local_item_size[0] , num_groups[1] * local_item_size[1] };

	// Alokacija pomnilnika na napravi
	cl_mem imageIn_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height*width * sizeof(unsigned char), imageIn, &ret);
	cl_mem imageOut_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, height*width * sizeof(unsigned char), NULL, &ret);

	// Priprava programa
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, NULL, &ret);
	// kontekst, "stevilo kazalcev na kodo, kazalci na kodo,		
	// stringi so NULL terminated, napaka													

	// Prevajanje
	ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);
	// program, "stevilo naprav, lista naprav, opcije pri prevajanju,
	// kazalec na funkcijo, uporabni"ski argumenti

	// Log
	size_t build_log_len;
	char* build_log;
	ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
	// program, "naprava, tip izpisa, 
	// maksimalna dol"zina niza, kazalec na niz, dejanska dol"zina niza
	build_log = (char*)malloc(sizeof(char) * (build_log_len + 1));
	ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, build_log_len, build_log, NULL);
	printf("%s\n", build_log);
	free(build_log);

	// "s"cepec: priprava objekta
	cl_kernel kernel = clCreateKernel(program, "sobelGPU", &ret);
	// program, ime "s"cepca, napaka

	size_t buf_size_t;
	clGetKernelWorkGroupInfo(kernel, device_id[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(buf_size_t), &buf_size_t, NULL);
	printf("veckratnik niti = %d", buf_size_t);

	scanf("%c", &ch);

	// "s"cepec: argumenti
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&imageIn_mem);
	ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&imageOut_mem);
	ret |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&width);
	ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&height);
	ret |= clSetKernelArg(kernel, 4, local_item_size[0] * local_item_size[1] * sizeof(unsigned char), NULL);
	// "s"cepec, "stevilka argumenta, velikost podatkov, kazalec na podatke

	// "s"cepec: zagon
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);
	// vrsta, "s"cepec, dimenzionalnost, mora biti NULL, 
	// kazalec na "stevilo vseh niti, kazalec na lokalno "stevilo niti, 
	// dogodki, ki se morajo zgoditi pred klicem
	// Kopiranje rezultatov
	BEGCLOCK(GPU)
	ret = clEnqueueReadBuffer(command_queue, imageOut_mem, CL_TRUE, 0, height * width * sizeof(unsigned char), imageOut, 0, NULL, NULL);
	ENDCLOCK(GPU)
	// branje v pomnilnik iz naparave, 0 = offset
	// zadnji trije - dogodki, ki se morajo zgoditi prej

	//Shrani sliko:
	FIBITMAP* imageOutBitmap = FreeImage_ConvertFromRawBits(imageOut, width, height, pitch, 8, 0xFF, 0xFF, 0xFF, TRUE);
	FreeImage_Save(FIF_PNG, imageOutBitmap, "sobel_slika_GPU.png", 0);
	FreeImage_Unload(imageOutBitmap);


	// "ci"s"cenje
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(imageIn_mem);
	ret = clReleaseMemObject(imageOut_mem);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(imageOut);
	free(imageIn);
}

int main(void){
	//CPU();
	GPU();
	return 0;
}

