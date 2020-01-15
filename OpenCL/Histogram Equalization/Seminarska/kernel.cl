#define BINS 256

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void HistogramGPU(__global unsigned char* imageIn, __global unsigned int* histogram, int width, int height) {

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

__kernel void CDF_GPU(__global unsigned int* histogram, __global unsigned int* cdf){
	
	/*nieve way: */
	/*cdf[0] = histogram[0];
	for (int i = 1; i < BINS; i++) {
		cdf[i] = cdf[i - 1] + histogram[i];
	} */

	int lid = get_local_id(0);
	int offset = 1;
	int n = BINS;

	__local unsigned int local_cdf[BINS + 1];

	local_cdf[lid] = histogram[lid];

	for (int d = n >> 1; d > 0; d >>= 1){
		__syncthreads();
		if (lid < d){
			int ai = offset * (2 * lid + 1) - 1;
			int bi = offset * (2 * lid + 2) - 1;
			local_cdf[bi] += local_cdf[ai];
		}
		offset *= 2;
	}
	if (lid == 0) { 
		local_cdf[n] = local_cdf[n - 1];
		local_cdf[n - 1] = 0;
	}
	for (int d = 1; d < n; d *= 2){
		offset >>= 1;
		__syncthreads();
		if (lid < d){
			int ai = offset * (2 * lid + 1) - 1;
			int bi = offset * (2 * lid + 2) - 1;
			float t = local_cdf[ai];
			local_cdf[ai] = local_cdf[bi];
			local_cdf[bi] += t;
		}
	}
	__syncthreads();

	//prvi korak so same 0-le.
	cdf[lid] = local_cdf[lid + 1];
}


inline unsigned int findMin(__global unsigned int* cdf) {
	/* ------------------------- Naiven nacin ------------------------------*/
	/*unsigned int min = 0;
	for (int i = 0; min == 0 && i < BINS; i++) {
		min = cdf[i];
	}
	return min;*/
	/* --------------------------- Reduction (WIP) -----------------------------*/
	//isto kot na predavanju 
	/*int local_y = get_local_id(1);
	int local_x = get_local_id(0);
	int local_index = local_y * get_local_size(0) + local_x;
	__local unsigned int cdf_local[BINS];
	if (local_index < BINS) cdf_local[local_index] = cdf[local_index];
	barrier(CLK_LOCAL_MEM_FENCE);

	int offset = BINS / 2;
	while (offset != 0) {
		if (local_index < offset) {
			int normal = cdf_local[local_index];
			int next = cdf_local[local_index + offset];
			if (normal < next)
				if(normal != 0)cdf_local[local_index] = normal;
				else cdf_local[local_index] = next;
			else {
				if (next != 0)cdf_local[local_index] = next;
				else cdf_local[local_index] = normal;
			}
		}
		offset = offset / 2;
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (local_index == 0) {
		return cdf_local[0];
	}*/
	

	/*---------------------------- Reduction 2 --------------------------------*/
	//isto kot pri cdfju z bit shiftom
	int local_x = get_local_id(1);
	int local_y = get_local_id(0);

	int local_index = local_x * get_local_size(0) + local_y;

	int offset = 1;

	__local unsigned long cdf_local[BINS];

	if (local_index < BINS)cdf_local[local_index] = cdf[local_index];

	barrier(CLK_LOCAL_MEM_FENCE);
	for (int d = BINS >> 1; d > 0; d >>= 1) {
		if (local_index < d) {
			int normal = offset * (2 * local_index + 1) - 1;
			int next = offset * (2 * local_index + 2) - 1;

			if (cdf_local[next] > cdf_local[normal] && cdf_local[normal] != 0) {
				cdf_local[next] = cdf_local[normal];
			}

		}
		offset *= 2;
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	return cdf_local[BINS-1];
	/* --------------------------- faster way -----------------------*/
	
	/*int local_x = get_local_id(1);
	int local_y = get_local_id(0);

	int local_index = local_x * get_local_size(0) + local_y;
	__local unsigned int min;

	if (local_index != 0 && cdf[local_index] != 0 && cdf[local_index - 1] == 0)min = cdf[local_index];
	barrier(CLK_LOCAL_MEM_FENCE);

	return min;*/
	/* --------------------------------------------------------------*/
}
inline unsigned char scale(unsigned long cdf, unsigned long cdfmin, unsigned long imageSize) {
	float scale;
	scale = (float)(cdf - cdfmin) / (float)(imageSize - cdfmin);
	scale = round(scale * (float)(BINS - 1));
	return (int)scale;
}


__kernel void EqualizeGPU(__global unsigned int *histogram, __global unsigned int *cdf, __global unsigned char* imageOut, int width, int height) {

	/* ---------------------------------- nieve way: ---------------------------------- */
	/*if (get_global_id(1) == 0 && get_global_id(0) == 0) {
		unsigned long imageSize = width * height;
		unsigned long cdfmin = findMin(cdf, histogram);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				imageOut[i * width + j] = scale(cdf[imageOut[i * width + j]], cdfmin, imageSize);
				//printf("[%d, %d]: %d\n", i, j, imageOut[i * width + j]);
			}
		}
	}
	*/

	/* ---------------------------------- parallel: ---------------------------------- */
	int i = get_global_id(0);
	int j = get_global_id(1);
	unsigned long imageSize = width * height;
	unsigned long cdfmin = findMin(cdf);
	if (get_global_id(1) == 0 && get_global_id(0) == 0) {
		printf("Min: %u\n", cdfmin);
	}

	if (j < width && i < height) {
		imageOut[i * width + j] = scale(cdf[imageOut[i * width + j]], cdfmin, imageSize);
	}
}

// @z33hun
