#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include "timing.h"
#include <math.h>
#include "FreeImage.h"
#include "CPU.h"

#define BINS 256
#define MAX_SOURCE_SIZE	16384

#define TEST true

#define BINS 256
#define image_file "slike/big.png"
//#define image_file "GPU_output.png"

struct seminarska {
	FIBITMAP* imageBitmap;
	FIBITMAP* imageBitmapGrey;
	int width;
	int height;
	char* source_str;
	size_t source_size;
	cl_device_id device_id[10];
	cl_int ret;
	cl_context context;
	cl_command_queue command_queue;
	cl_program program;

	//histogram:
	unsigned char* imageRaw;
	unsigned long* histogram;
	cl_mem image_mem_object;
	cl_mem histogram_ret_object;

	//cdf:
	unsigned long* CDF;
	cl_mem cdf_ret_object;

	//time
	double GPU_time;
	clock_t begclock;
	clock_t endclock;
} sem;


void init() {
	sem.GPU_time = 0;
	sem.begclock = clock();
	// set the image
	sem.imageBitmap = FreeImage_Load(FIF_PNG, image_file, 0);
	sem.imageBitmapGrey = FreeImage_ConvertToGreyscale(sem.imageBitmap);
	sem.width = FreeImage_GetWidth(sem.imageBitmapGrey);
	sem.height = FreeImage_GetHeight(sem.imageBitmapGrey);
	//read the kernel
	FILE* fp;
	fp = fopen("kernel.cl", "r");
	if (!fp){
		fprintf(stderr, ":-(#\n");
		exit(1);
	}
	sem.source_str = (char*)malloc(MAX_SOURCE_SIZE);
	sem.source_size = fread(sem.source_str, 1, MAX_SOURCE_SIZE, fp);
	sem.source_str[sem.source_size] = '\0';
	fclose(fp);
	sem.endclock = clock();
	sem.GPU_time += (((double)(sem.endclock - sem.begclock)) / CLOCKS_PER_SEC); //-----------------------------------------------------------------------------time for kernel + image read

	cl_platform_id	platform_id[10];
	cl_uint			ret_num_platforms;
	size_t			buf_len;
	sem.ret  = clGetPlatformIDs(10, platform_id, &ret_num_platforms);
	// Podatki o napravi
	cl_uint			ret_num_devices;
	// Delali bomo s platform_id[0] na GPU
	sem.ret = clGetDeviceIDs(platform_id[0], CL_DEVICE_TYPE_GPU, 10, sem.device_id, &ret_num_devices);

	// Kontekst
	sem.context = clCreateContext(NULL, 1, &sem.device_id[0], NULL, NULL, &sem.ret);

	// Ukazna vrsta
	sem.command_queue = clCreateCommandQueue(sem.context, sem.device_id[0], 0, &sem.ret);

	// Priprava programa
	sem.program = clCreateProgramWithSource(sem.context, 1, (const char**)&sem.source_str, NULL, &sem.ret);
	// kontekst, "stevilo kazalcev na kodo, kazalci na kodo,		
	// stringi so NULL terminated, napaka

	sem.begclock = clock(); 
	// Prevajanje
	sem.ret = clBuildProgram(sem.program, 1, &sem.device_id[0], NULL, NULL, NULL);
	// Log
	size_t build_log_len;
	char* build_log;
	sem.ret = clGetProgramBuildInfo(sem.program, sem.device_id[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_len);
	// program, naprava, tip izpisa, 
	// maksimalna dolzina niza, kazalec na niz, dejanska dolzina niza

	build_log = (char*)malloc(sizeof(char) * (build_log_len + 1));
	sem.ret = clGetProgramBuildInfo(sem.program, sem.device_id[0], CL_PROGRAM_BUILD_LOG, build_log_len, build_log, NULL);
	printf("%s\n", build_log);
	free(build_log);
	// set the variables to histogramGPU ----------------------------------------------------------------
	sem.imageRaw = (unsigned char*)malloc(sem.height * sem.width * sizeof(unsigned long));
	FreeImage_ConvertToRawBits(sem.imageRaw, sem.imageBitmapGrey, sem.width, 8, 0xFF, 0xFF, 0xFF, TRUE);

	sem.histogram = (unsigned long*)malloc(BINS * sizeof(unsigned int));
	memset(sem.histogram, 0, BINS * sizeof(unsigned long));

	// Alokacija pomnilnika na napravi
	sem.image_mem_object = clCreateBuffer(sem.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sem.height * sem.width * sizeof(unsigned char), sem.imageRaw, &sem.ret);
	sem.histogram_ret_object = clCreateBuffer(sem.context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, BINS * sizeof(unsigned long), sem.histogram, &sem.ret);


	// set the variables to CDFGPU ----------------------------------------------------------------------
	sem.CDF = (unsigned long*)malloc(BINS * sizeof(unsigned long));
	memset(sem.CDF, 0, BINS * sizeof(unsigned long));
	sem.endclock = clock(); 
	sem.GPU_time += (((double)(sem.endclock - sem.begclock)) / CLOCKS_PER_SEC); // -------------------------------------------------------------------------- time for initing the memory
}

// ---------------------------------------------------------------------------  HISTOGRAM -----------------------------------------------------------------------------------------------
void GPUHistogram() {

	cl_kernel kernel = clCreateKernel(sem.program, "HistogramGPU", &sem.ret);
	size_t buf_size_t;
	clGetKernelWorkGroupInfo(kernel, sem.device_id[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(buf_size_t), &buf_size_t, NULL);
	printf("veckratnik niti = %d\n", buf_size_t);

	printf("Width: %d\n", sem.width);
	printf("Hieght: %d\n\n", sem.height);

	// Delitev dela
	size_t local_item_size[2] = { 32, 32 };
	size_t num_groups[2] = { (sem.height - 1) / local_item_size[0] + 1 , (sem.width - 1) / local_item_size[1] + 1 };
	size_t global_item_size[2] = { num_groups[0] * local_item_size[0] , num_groups[1] * local_item_size[1] };

	// argumenti
	sem.ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&sem.image_mem_object);
	sem.ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&sem.histogram_ret_object);
	sem.ret |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&sem.width);
	sem.ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&sem.height);

	sem.ret = clEnqueueNDRangeKernel(sem.command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);

	// Kopiranje rezultatov
	sem.begclock = clock();
	sem.ret = clEnqueueReadBuffer(sem.command_queue, sem.histogram_ret_object, CL_TRUE, 0, BINS * sizeof(unsigned long), sem.histogram, 0, NULL, NULL);
	sem.endclock = clock();
	sem.GPU_time += (((double)(sem.endclock - sem.begclock)) / CLOCKS_PER_SEC); // -------------------------------------------------------------------------- time for running HistogramGPU

}

// --------------------------------------------------------------------------------  CDF -----------------------------------------------------------------------------------------------
void GPU_CDF() {

	// Alokacija pomnilnika na napravi
	sem.begclock = clock();
	cl_mem histogram_object = clCreateBuffer(sem.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, BINS * sizeof(unsigned long), sem.histogram, &sem.ret);
	sem.cdf_ret_object = clCreateBuffer(sem.context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, BINS * sizeof(unsigned long), sem.CDF, &sem.ret);
	sem.endclock = clock();
	sem.GPU_time += (((double)(sem.endclock - sem.begclock)) / CLOCKS_PER_SEC);  //--------------------------------------------------------------------------- time for initing the memory
	
	//priprava objekta
	cl_kernel kernel = clCreateKernel(sem.program, "CDF_GPU", &sem.ret);

	size_t buf_size_t;
	clGetKernelWorkGroupInfo(kernel, sem.device_id[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(buf_size_t), &buf_size_t, NULL);

	size_t local_item_size = BINS;
	size_t num_groups = (BINS - 1) / local_item_size + 1;
	size_t global_item_size = local_item_size * num_groups;

	sem.ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&histogram_object);
	sem.ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&sem.cdf_ret_object);

	// zagon
	sem.ret = clEnqueueNDRangeKernel(sem.command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

	sem.begclock = clock();

	// Kopiranje rezultatov
	sem.ret = clEnqueueReadBuffer(sem.command_queue, sem.cdf_ret_object, CL_TRUE, 0, BINS * sizeof(unsigned long), sem.CDF, 0, NULL, NULL);
	sem.endclock = clock();
	sem.GPU_time += (((double)(sem.endclock - sem.begclock)) / CLOCKS_PER_SEC); // ---------------------------------------------------------------------------time for running CDFGPU
}


// ---------------------------------------------------------------------------  EQUALIZE -----------------------------------------------------------------------------------------------



void GPU_Equalize() {
	sem.begclock = clock();
	cl_mem histogram_ret_object = clCreateBuffer(sem.context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, BINS * sizeof(unsigned long), sem.histogram, &sem.ret);
	sem.endclock = clock();
	sem.GPU_time += (((double)(sem.endclock - sem.begclock)) / CLOCKS_PER_SEC); // --------------------------------------------------------------------------- time for initing memory

	cl_kernel kernel = clCreateKernel(sem.program, "EqualizeGPU", &sem.ret);
	size_t buf_size_t;
	clGetKernelWorkGroupInfo(kernel, sem.device_id[0], CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(buf_size_t), &buf_size_t, NULL);

	size_t local_item_size[2] = { 32, 32 };
	size_t num_groups[2] = { (sem.height - 1) / local_item_size[0] + 1 , (sem.width - 1) / local_item_size[1] + 1 };
	size_t global_item_size[2] = { num_groups[0] * local_item_size[0] , num_groups[1] * local_item_size[1] };

	//argumenti
	sem.ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&histogram_ret_object);
	sem.ret |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&sem.cdf_ret_object);
	sem.ret |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&sem.image_mem_object);
	sem.ret |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&sem.width);
	sem.ret |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&sem.height);

	//zagon
	sem.ret = clEnqueueNDRangeKernel(sem.command_queue, kernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);

	sem.begclock = clock();

	// Kopiranje rezultatov
	sem.ret = clEnqueueReadBuffer(sem.command_queue, sem.image_mem_object, CL_TRUE, 0, sem.height * sem.width * sizeof(unsigned char), sem.imageRaw, 0, NULL, NULL);
	

	/* Zapisi image */
	FIBITMAP* imageOutBitmap = FreeImage_ConvertFromRawBits(sem.imageRaw, sem.width, sem.height, sem.width, 8, 0xFF, 0xFF, 0xFF, TRUE);
	// Save image to file
	FreeImage_Save(FIF_PNG, imageOutBitmap, "GPU_output.png", 0);
	sem.endclock = clock();
	sem.GPU_time += (((double)(sem.endclock - sem.begclock)) / CLOCKS_PER_SEC); // --------------------------------------------------------------- time for running Equalize and image save
	//clear
	sem.ret = clReleaseKernel(kernel);
	sem.ret = clReleaseMemObject(sem.image_mem_object);
	sem.ret = clReleaseMemObject(histogram_ret_object);
}

void clear() {
	sem.ret = clReleaseMemObject(sem.cdf_ret_object);
	sem.ret = clFlush(sem.command_queue);
	sem.ret = clFinish(sem.command_queue);
	sem.ret = clReleaseProgram(sem.program);
	sem.ret = clReleaseCommandQueue(sem.command_queue);
	sem.ret = clReleaseContext(sem.context);
	FreeImage_Unload(sem.imageBitmapGrey);
	FreeImage_Unload(sem.imageBitmap);
}

void GPU() {
	init();
	GPUHistogram();
	GPU_CDF();
	GPU_Equalize();
}

int main(void) {
	unsigned char* CPU_img = (unsigned char*)malloc(sem.height * sem.width * sizeof(unsigned long));
	BEGCLOCK(CPU_ALL);
	CPU_img = CPU();
	ENDCLOCK(CPU_ALL);
	GPU();
	printf("CLOCK(GPU_ALL): %lf\n", sem.GPU_time);
	if (TEST) {
		int error = 0;
		for (int i = 0; i < sem.height * sem.width; i++) {
			if ((int)CPU_img[i] != (int)sem.imageRaw[i]) {
				printf("[%d] CPU: %d - GPU: %d\n", i, CPU_img[i], sem.imageRaw[i]); error++;
			}
		}
		if (error > 0)printf("Napaka!");
		else printf("CPU in GPU se ujemata!");
	}
	return 0;
}