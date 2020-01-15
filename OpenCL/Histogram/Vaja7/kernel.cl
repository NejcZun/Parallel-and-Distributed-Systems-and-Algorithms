#define BINS 256
__kernel void GlobalHistogramGPU(__global unsigned char* imageIn, __global unsigned int* histogram, int width, int height) {

	//global indexes
	int gidx = get_global_id(1);
	int gidy = get_global_id(0);
	int global_index = gidy * width + gidx;
	if (gidx < width && gidy < height) {
		unsigned int pixel = imageIn[global_index];
		atom_add(&histogram[pixel], 1);
	}

}

__kernel void LocalHistogramGPU(__global unsigned char* imageIn, __global unsigned int* histogram, int width, int height) {

	//global indexes
	int gidx = get_global_id(1);
	int gidy = get_global_id(0);
	int global_index = gidy * width + gidx;
	
	//local indexes
	__local unsigned int local_hist[BINS];
	int lidx = get_local_id(1);
	int lidy = get_local_id(0);
	int local_index = lidy * get_local_size(0) + lidx;
	
	int rowGroups = get_group_id(1);
	int colGroups = get_group_id(0);

	unsigned int pixel;
	
	//nastav local histogram na 0
	if (local_index < BINS) local_hist[local_index] = 0;
	barrier(CLK_LOCAL_MEM_FENCE);
	if ((lidx + colGroups * get_local_size(0)) < height && (lidy + rowGroups * get_local_size(0) < width)) {
		pixel = imageIn[(lidx + colGroups * get_local_size(0)) * width + (lidy + rowGroups * get_local_size(0))];
		atom_add(&local_hist[pixel], 1);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	//fill the histogram with the temp histogram
	if (local_index < BINS) atom_add(&histogram[local_index], local_hist[local_index]);

}