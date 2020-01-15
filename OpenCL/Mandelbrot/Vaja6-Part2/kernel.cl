__kernel void mandelbrotGPU(__global unsigned char* image, int width, int height) {

	int i = get_global_id(0);
	int j = get_global_id(1);

	if (j < width && i < height) {
		float x0, y0, x, y, xtemp;
		int color;
		int iter;
		int max_iteration = 800;   //max stevilo iteracij
		unsigned char max = 255;   //max vrednost barvnega kanala

		x0 = (float)j / width * (float)3.5 - (float)2.5; //zacetna vrednost
		y0 = (float)i / height * (float)2.0 - (float)1.0;
		x = 0;
		y = 0;
		iter = 0;
		//ponavljamo, dokler ne izpolnemo enega izmed pogojev
		while ((x * x + y * y <= 4) && (iter < max_iteration))
		{
			xtemp = x * x - y * y + x0;
			y = 2 * x * y + y0;
			x = xtemp;
			iter++;
		}
		//izracunamo barvo (magic: http://linas.org/art-gallery/escape/smooth.html)
		color = 1.0 + iter - log(log(sqrt(x * x + y * y))) / log(2.0);
		color = (8 * max * color) / max_iteration;
		if (color > max)
			color = max;
		//zapisemo barvo RGBA (v resnici little endian BGRA)
		image[4 * i * width + 4 * j + 0] = color; //Blue
		image[4 * i * width + 4 * j + 1] = color; // Green
		image[4 * i * width + 4 * j + 2] = color; // Red
		image[4 * i * width + 4 * j + 3] = 255;   // Alpha
	}
}