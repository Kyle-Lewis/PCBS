#include "PCBS.h"
#include <math.h>

#define PI 3.14159265

int main()
{
	int width = 128;
	int height = 128;
	int nFrames = 11;

	float* frames = (float*)malloc(nFrames*width*height*sizeof(float));
	#define pidx width*height*fidx + height*row + col
	for (int fidx = 0; fidx < nFrames; fidx++)
	{
		for (int row = 0; row < height; row++)
		{
			for (int col = 0; col < width; col++)
			{
				frames[width*height*fidx + height*row + col] = sin(col*PI/180.0);
			}
		}
	}

	int numPoints = 16;
	int* indices = (int*)malloc(numPoints*sizeof(int));
	indices[0] = width * (height - 64) + 5;
	indices[1] = width * (height - 64) + 6;
	indices[2] = width * (height - 64) + 7;
	indices[3] = width * (height - 64) + 8;

	indices[4] = width * (height - 63) + 5;
	indices[5] = width * (height - 63) + 6;
	indices[6] = width * (height - 63) + 7;
	indices[7] = width * (height - 63) + 8;

	indices[8] = width * (height - 62) + 5;
	indices[9] = width * (height - 62) + 6;
	indices[10] = width * (height - 62) + 7;
	indices[11] = width * (height - 62) + 8;

	indices[12] = width * (height - 61) + 5;
	indices[13] = width * (height - 61) + 6;
	indices[14] = width * (height - 61) + 7;
	indices[15] = width * (height - 61) + 8;

	for (int fidx = 0; fidx < nFrames; fidx++)
	{
		for (int i = 0; i < numPoints; i++)
		{
			frames[indices[i] + 2*fidx] = 0.0;
		}
	}

	float* testFrame = frames + 10*width*height*sizeof(float);

	Static_PCBS(width, height, nFrames-1, frames, testFrame);

	free(indices);
	free(frames);
	return 0;

} 