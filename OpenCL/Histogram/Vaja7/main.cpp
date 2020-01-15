#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "timing.h"
#include <math.h>
#include "FreeImage.h"

#define BINS 256
#define MAX_SOURCE_SIZE	16384
#define MAX(a,b) (((a)>(b))?(a):(b))
#define RUN "LOCAL"
#define BINS 256
#define image "yue.png" //256 x 256 


void HistogramCPU(unsigned char* imageIn, unsigned int* histogram, int width, int height){

	memset(histogram, 0, BINS * sizeof(unsigned int));

	//za vsak piksel v sliki
	for (int i = 0; i < (height); i++)
		for (int j = 0; j < (width); j++){
			histogram[imageIn[i * width + j]]++;
		}
}

void printHistogram(unsigned int* histogram) {
	printf("Barva\tPojavitve\n");
	for (int i = 0; i < BINS; i++) {
		printf("%d\t%d\n", i, histogram[i]);
	}
}

unsigned int *CPU(void){

	FIBITMAP* imageBitmap = FreeImage_Load(FIF_PNG, image, 0);
	FIBITMAP* imageBitmapGrey = FreeImage_ConvertToGreyscale(imageBitmap);
	int width = FreeImage_GetWidth(imageBitmapGrey);
	int height = FreeImage_GetHeight(imageBitmapGrey);
	int pitch = FreeImage_GetPitch(imageBitmapGrey);

	unsigned char* imageIn = (unsigned char*)malloc(height * width * sizeof(unsigned char));
	unsigned int* histogram = (unsigned int*)malloc(BINS * sizeof(unsigned int));

	FreeImage_ConvertToRawBits(imageIn, imageBitmapGrey, pitch, 8, 0xFF, 0xFF, 0xFF, TRUE);

	FreeImage_Unload(imageBitmapGrey);
	FreeImage_Unload(imageBitmap);

	BEGCLOCK(CPU)
	HistogramCPU(imageIn, histogram, width, height);
	ENDCLOCK(CPU)
	printHistogram(histogram);

	return histogram;
}
unsigned int *GPU() {
	FIBITMAP* imageBitmap = FreeImage_Load(FIF_PNG, image, 0);
	FIBITMAP* imageBitmapGrey = FreeImage_ConvertToGreyscale(imageBitmap);
	int width = FreeImage_GetWidth(imageBitmapGrey);
	int height = FreeImage_GetHeight(imageBitmapGrey);
	int pitch = FreeImage_GetPitch(imageBitmapGrey);

	unsigned char* imageIn = (unsigned char*)malloc(height * width * sizeof(unsigned char));
	unsigned int* histogram = (unsigned int*)malloc(BINS * sizeof(unsigned int));
	memset(histogram, 0, BINS * sizeof(unsigned int));

	FreeImage_ConvertToRawBits(imageIn, imageBitmapGrey, pitch, 8, 0xFF, 0xFF, 0xFF, TRUE);

	FreeImage_Unload(imageBitmapGrey);
	FreeImage_Unload(imageBitmap);

	if (RUN == "GLOBAL") {
		// send it
		char ch;
		int i;
		cl_int ret;

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

		// Rezervacija pomnilnika
		//unsigned char* imageGPU = (unsigned char*)malloc(height * width * sizeof(unsigned char) * 4);

		// Podatki o platformi
		cl_platform_id	platform_id[10];
		cl_uint			ret_num_platforms;
		char* buf;
		size_t			buf_len;
		ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);
		// max. "stevilo platform, kazalec na platforme, dejansko "stevilo platform

		// Podatki o napravi
		cl_device_id	device_id[10];
		cl_uint			ret_num_devices;
		// Delali bomo s platform_id[0] na GPU
		ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10,
			device_id, &ret_num_devices);

		// Kontekst
		cl_context context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &ret);
		// kontekst: vklju"cene platforme - NULL je privzeta, "stevilo naprav, 
		// kazalci na naprave, kazalec na call-back funkcijo v primeru napake
		// dodatni parametri funkcije, "stevilka napake

		// Ukazna vrsta
		cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);
		// kontekst, naprava, INORDER/OUTOFORDER, napake

		// Alokacija pomnilnika na napravi
		cl_mem image_mem_object = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			height * width * sizeof(unsigned char), imageIn, &ret);


		cl_mem histogram_ret_object = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
			BINS * sizeof(unsigned int), histogram, &ret);

		// Priprava programa
		cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str,
			NULL, &ret);
		// kontekst, "stevilo kazalcev na kodo, kazalci na kodo,		
		// stringi so NULL terminated, napaka													

		// Prevajanje
		ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);
		// program, "stevilo naprav, lista naprav, opcije pri prevajanju,
		// kazalec na funkcijo, uporabni"ski argumenti

		// Log
		size_t build_log_len;
		char* build_log;
		ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
			0, NULL, &build_log_len);
		// program, "naprava, tip izpisa, 
		// maksimalna dol"zina niza, kazalec na niz, dejanska dol"zina niza
		build_log = (char*)malloc(sizeof(char) * (build_log_len + 1));
		ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
			build_log_len, build_log, NULL);
		printf("%s\n", build_log);
		free(build_log);

		// "s"cepec: priprava objekta
		cl_kernel kernel = clCreateKernel(program, "GlobalHistogramGPU", &ret);
		//cl_kernel kernel = clCreateKernel(program, "histogram_local", &ret);
		// program, ime "s"cepca, napaka

		size_t buf_size_t;
		clGetKernelWorkGroupInfo(kernel, device_id[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(buf_size_t), &buf_size_t, NULL);
		printf("veckratnik niti = %d", buf_size_t);

		scanf("%c", &ch);


		printf("Width: %d\n", width);
		printf("Hieght: %d\n\n", height);

		// Delitev dela
		//size_t local_item_size[2] = {32, 32};
		//size_t num_groups = ((vectorSize - 1) / local_item_size + 1);
		//size_t global_item_size[2] = {width, height};

		//size_t local_item_size[2] = { 16, 16 };
		size_t local_item_size[2] = { 32, 32 };
		size_t num_groups[2] = { (height - 1) / local_item_size[0] + 1 , (width - 1) / local_item_size[1] + 1 };
		size_t global_item_size[2] = { num_groups[0] * local_item_size[0] , num_groups[1] * local_item_size[1] };


		// "s"cepec: argumenti
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&image_mem_object);
		ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&histogram_ret_object);
		ret |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&width);
		ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&height);
		// "s"cepec, "stevilka argumenta, velikost podatkov, kazalec na podatke

		// "s"cepec: zagon
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
			global_item_size, local_item_size, 0, NULL, NULL);
		// vrsta, "s"cepec, dimenzionalnost, mora biti NULL, 
		// kazalec na "stevilo vseh niti, kazalec na lokalno "stevilo niti, 
		// dogodki, ki se morajo zgoditi pred klicem

		BEGCLOCK(gpu_time);

		// Kopiranje rezultatov

		ret = clEnqueueReadBuffer(command_queue, histogram_ret_object, CL_TRUE, 0,
			BINS * sizeof(unsigned int), histogram, 0, NULL, NULL);


		// branje v pomnilnik iz naparave, 0 = offset
		// zadnji trije - dogodki, ki se morajo zgoditi prej

		// Prikaz rezultatov
		printHistogram(histogram);

		ENDCLOCK(gpu_time);

		// "ci"s"cenje
		ret = clFlush(command_queue);
		ret = clFinish(command_queue);
		ret = clReleaseKernel(kernel);
		ret = clReleaseProgram(program);
		ret = clReleaseMemObject(image_mem_object);
		ret = clReleaseMemObject(histogram_ret_object);
		ret = clReleaseCommandQueue(command_queue);
		ret = clReleaseContext(context);

		free(imageIn);

	}
	else {
		// send it
		char ch;
		int i;
		cl_int ret;

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

		// Rezervacija pomnilnika
		//unsigned char* imageGPU = (unsigned char*)malloc(height * width * sizeof(unsigned char) * 4);

		// Podatki o platformi
		cl_platform_id	platform_id[10];
		cl_uint			ret_num_platforms;
		char* buf;
		size_t			buf_len;
		ret = clGetPlatformIDs(10, platform_id, &ret_num_platforms);
		// max. "stevilo platform, kazalec na platforme, dejansko "stevilo platform

		// Podatki o napravi
		cl_device_id	device_id[10];
		cl_uint			ret_num_devices;
		// Delali bomo s platform_id[0] na GPU
		ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10,
			device_id, &ret_num_devices);

		// Kontekst
		cl_context context = clCreateContext(NULL, 1, &device_id[0], NULL, NULL, &ret);
		// kontekst: vklju"cene platforme - NULL je privzeta, "stevilo naprav, 
		// kazalci na naprave, kazalec na call-back funkcijo v primeru napake
		// dodatni parametri funkcije, "stevilka napake

		// Ukazna vrsta
		cl_command_queue command_queue = clCreateCommandQueue(context, device_id[0], 0, &ret);
		// kontekst, naprava, INORDER/OUTOFORDER, napake

		// Alokacija pomnilnika na napravi
		cl_mem image_mem_object = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, height * width * sizeof(unsigned char), imageIn, &ret);


		cl_mem histogram_ret_object = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, BINS * sizeof(unsigned int), histogram, &ret);

		// Priprava programa
		cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str,
			NULL, &ret);
		// kontekst, "stevilo kazalcev na kodo, kazalci na kodo,		
		// stringi so NULL terminated, napaka													

		// Prevajanje
		ret = clBuildProgram(program, 1, &device_id[0], NULL, NULL, NULL);
		// program, "stevilo naprav, lista naprav, opcije pri prevajanju,
		// kazalec na funkcijo, uporabni"ski argumenti

		// Log
		size_t build_log_len;
		char* build_log;
		ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
			0, NULL, &build_log_len);
		// program, "naprava, tip izpisa, 
		// maksimalna dol"zina niza, kazalec na niz, dejanska dol"zina niza
		build_log = (char*)malloc(sizeof(char) * (build_log_len + 1));
		ret = clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG,
			build_log_len, build_log, NULL);
		printf("%s\n", build_log);
		free(build_log);

		// "s"cepec: priprava objekta
		cl_kernel kernel = clCreateKernel(program, "LocalHistogramGPU", &ret);
		//cl_kernel kernel = clCreateKernel(program, "histogram_local", &ret);
		// program, ime "s"cepca, napaka

		size_t buf_size_t;
		clGetKernelWorkGroupInfo(kernel, device_id[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(buf_size_t), &buf_size_t, NULL);
		printf("veckratnik niti = %d", buf_size_t);

		scanf("%c", &ch);


		printf("Width: %d\n", width);
		printf("Hieght: %d\n\n", height);

		// Delitev dela
		//size_t local_item_size[2] = {32, 32};
		//size_t num_groups = ((vectorSize - 1) / local_item_size + 1);
		//size_t global_item_size[2] = {width, height};

		//size_t local_item_size[2] = { 16, 16 };
		size_t local_item_size[2] = { 32, 32 };
		size_t num_groups[2] = { (height - 1) / local_item_size[0] + 1 , (width - 1) / local_item_size[1] + 1 };
		size_t global_item_size[2] = { num_groups[0] * local_item_size[0] , num_groups[1] * local_item_size[1] };

		// "s"cepec: argumenti
		ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&image_mem_object);
		ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&histogram_ret_object);
		ret |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&width);
		ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&height);
		// "s"cepec, "stevilka argumenta, velikost podatkov, kazalec na podatke

		// "s"cepec: zagon
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL,
			global_item_size, local_item_size, 0, NULL, NULL);
		// vrsta, "s"cepec, dimenzionalnost, mora biti NULL, 
		// kazalec na "stevilo vseh niti, kazalec na lokalno "stevilo niti, 
		// dogodki, ki se morajo zgoditi pred klicem

		BEGCLOCK(local_GPU);

		// Kopiranje rezultatov

		ret = clEnqueueReadBuffer(command_queue, histogram_ret_object, CL_TRUE, 0, BINS * sizeof(unsigned int), histogram, 0, NULL, NULL);


		// branje v pomnilnik iz naparave, 0 = offset
		// zadnji trije - dogodki, ki se morajo zgoditi prej

		// Prikaz rezultatov
		printHistogram(histogram);

		ENDCLOCK(local_GPU);

		// "ci"s"cenje
		ret = clFlush(command_queue);
		ret = clFinish(command_queue);
		ret = clReleaseKernel(kernel);
		ret = clReleaseProgram(program);
		ret = clReleaseMemObject(image_mem_object);
		ret = clReleaseMemObject(histogram_ret_object);
		ret = clReleaseCommandQueue(command_queue);
		ret = clReleaseContext(context);

		free(imageIn);
		}
	return histogram;
}
int main(void){
	unsigned int* histogram_CPU = (unsigned int*)malloc(BINS * sizeof(unsigned int));
	unsigned int* histogram_GPU = (unsigned int*)malloc(BINS * sizeof(unsigned int));
	histogram_CPU = CPU();
	histogram_GPU = GPU();
	int error = 0;
	for (int i = 0; i < 256; i++) if (histogram_CPU[i] != histogram_GPU[i]) {
		printf("i: %d, CPU: %d, GPU: %d\n", i, histogram_CPU[i], histogram_GPU[i]);
		error++;
	}
	if (error > 0) printf("Ne dela!\n");
	else printf("Dela!\n");
	return 0;
}