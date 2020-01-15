inline int getPixel(__local unsigned char chunk[18][18], int width, int height, int y, int x)
{
	if (x < 0 || x > 17)
		return 0;
	if (y < 0 || y > 17)
		return 0;
	return chunk[x][y];
}

__kernel void sobelGPU(__global unsigned char* imageIn, __global unsigned char* imageOut, int width, int height, __local unsigned char chunk[18][18]) {

	//global indexes
	int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int global_index = gidy * width + gidx;
	int gW = get_num_groups(0) * get_local_size(0); // Width
	int gH = get_num_groups(1) * get_local_size(1); // Height
	
	//local indexes
	int lidx = get_local_id(0);
	int lidy = get_local_id(1);
	int chunk_size = 17;
	int size_x = get_local_size(0) - 1;
	int size_y = get_local_size(1) - 1;

	int Gx, Gy;
	int tempPixel;

	/*
		0 1 2 3 4 5 6 7 8 9 10 11 12 14 15 16
		1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
		2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
		3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
		4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
		5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
		6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
		7 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
		8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
		9 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
	   10 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
	   11 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
	   12 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
	   13 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
	   15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
	   16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
	*/

	chunk[lidx+1][lidy+1] = imageIn[global_index];

	if (lidx == 0 && lidy == 0 && gidx > 0 && gidy > 0) chunk[0][0] = imageIn[global_index-width-1]; 
	if (lidx == 0 && lidy == 15 && gidx > 0 && gidy < gH) chunk[0][chunk_size] = imageIn[global_index+width-1];
	if (lidx == 0 && gidx > 0) chunk[0][lidy+1] = imageIn[global_index - 1];
	if (lidy == 0 && gidy > 0) chunk[lidx+1][0] = imageIn[global_index - width];
	if (lidx == size_x && lidy == 0 && gidx < gW && gidy > 0) chunk[chunk_size][0] = imageIn[global_index-width+1];
	if (lidx == size_x && lidy == size_y && gidx < gW && gidy < gH) chunk[chunk_size][chunk_size] = imageIn[global_index+width+1];
	if (lidx == size_x && gidx < gW) chunk[chunk_size][lidy+1] = imageIn[global_index+1];
	if (lidy == size_y && gidy < gH) chunk[lidx+1][chunk_size] = imageIn[global_index + width];

	barrier(CLK_LOCAL_MEM_FENCE);

	int x = lidx + 1;
	int y = lidy + 1;

	Gx = -getPixel(chunk, width, height, y - 1, x - 1) - 2 * getPixel(chunk, width, height, y - 1, x) - getPixel(chunk, width, height, y - 1, x + 1) + getPixel(chunk, width, height, y + 1, x - 1) + 2 * getPixel(chunk, width, height, y + 1, x) + getPixel(chunk, width, height, y + 1, x + 1);
	Gy = -getPixel(chunk, width, height, y - 1, x - 1) - 2 * getPixel(chunk, width, height, y, x - 1) - getPixel(chunk, width, height, y + 1, x - 1) + getPixel(chunk, width, height, y - 1, x + 1) + 2 * getPixel(chunk, width, height, y, x + 1) + getPixel(chunk, width, height, y + 1, x + 1);
	tempPixel = sqrt((float)(Gx * Gx + Gy * Gy));

	if (tempPixel > 255)
		imageOut[global_index] = 255;
	else
		imageOut[global_index] = tempPixel;
	barrier(CLK_LOCAL_MEM_FENCE);
}