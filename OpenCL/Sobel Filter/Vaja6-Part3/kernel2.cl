inline int getPixel(__local unsigned char chunk[18][18], int width, int height, int y, int x)
{
	if (x < 0 || x > 17)
		return 0;
	if (y < 0 || y > 17)
		return 0;
	return chunk[x][y];
}


__kernel void sobelGPU(__global unsigned char* imageIn, __global unsigned char* imageOut, int width, int height, __local unsigned char chunk[18][18]) {
	
	int Gx, Gy, global_index;

	int gidx = get_global_id(0);
	int gidy = get_global_id(1);

	int lidx = get_local_id(0);
	int lidy = get_local_id(1);

	int gW = get_num_groups(0) * get_local_size(0);
	int gH = get_num_groups(1) * get_local_size(1);

	int tempPixel;
	global_index = gidx * width + gidy;
	//za vsak piksel v sliki
	//printf("Gx: %d, Gy: %d, gw:%d, gh:%d\n", gidx, gidy, gW, gH);
	//printf("Lx: %d, Ly: %d\n", lidx, lidy);
	chunk[lidx+1][lidy+1] = imageIn[global_index];
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
	if (lidx == 0 && gidx > 0) chunk[0][lidy + 1] = imageIn[global_index]; //u
	if (lidx == 0 && gidx == 0) chunk[0][lidy + 1] = imageIn[global_index]; //u middle
	if (lidx == 0 && lidy == 0 && gidx > 0 && gidy > 0)chunk[0][0] = imageIn[global_index - width - 1]; 
	if (lidx == 0 && lidy == 15 && gidx > 0 && gidy < gH) chunk[0][sizeof(chunk[0]) - 1] = imageIn[global_index + width - 1];
	if (lidy == 0 && gidx > 0) chunk[lidx + 1][0] = imageIn[global_index - width];
	if (lidx == 15 && lidy == 0 && gidx < gW && gidy > 0) chunk[sizeof(chunk[0]) - 1][0] = imageIn[global_index - width]; //r
	if (lidy == 15 && gidy < gH) chunk[lidx + 1][sizeof(chunk[0]) - 1] = imageIn[global_index + width];
	if (lidx == 15 && lidy == 15 && gidx < gW && gidy < gH) chunk[sizeof(chunk[0]) - 1][sizeof(chunk[0]) - 1] = imageIn[global_index + width]; //r
	if (lidx == 15 && gidx < gW) chunk[sizeof(chunk[0]) - 1][lidy + 1] = imageIn[global_index]; //r
	barrier(CLK_LOCAL_MEM_FENCE);
	
	int x = lidx + 1;
	int y = lidy + 1;

	Gx = -getPixel(chunk, width, height, y - 1, x - 1) - 2 * getPixel(chunk, width, height, y - 1, x) -
		getPixel(chunk, width, height, y - 1, x + 1) + getPixel(chunk, width, height, y + 1, x - 1) +
		2 * getPixel(chunk, width, height, y + 1, x) + getPixel(chunk, width, height, y + 1, x + 1);
	Gy = -getPixel(chunk, width, height, y - 1, x - 1) - 2 * getPixel(chunk, width, height, y, x - 1) -
		getPixel(chunk, width, height, y + 1, x - 1) + getPixel(chunk, width, height, y - 1, x + 1) +
		2 * getPixel(chunk, width, height, y, x + 1) + getPixel(chunk, width, height, y + 1, x + 1);
	tempPixel = sqrt((float)(Gx * Gx + Gy * Gy));
	
	if (tempPixel > 255) imageOut[global_index] = 255;
	else imageOut[global_index] = tempPixel;
	barrier(CLK_LOCAL_MEM_FENCE);
}