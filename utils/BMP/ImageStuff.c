#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "ImageStuff.h"

/*
 * Load a BMP image
 */

pel** ReadBMP(char* filename) {
	FILE* f = fopen(filename, "rb");
	if (f == NULL) {
		printf("\n\n%s NOT FOUND\n\n", filename);
		exit(1);
	}

	pel HeaderInfo[54];
	fread(HeaderInfo, sizeof(pel), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*) &HeaderInfo[18];
	int height = *(int*) &HeaderInfo[22];

	//copy header for re-use
	for (unsigned int i = 0; i < 54; i++)
		ip.HeaderInfo[i] = HeaderInfo[i];

	ip.Vpixels = height;
	ip.Hpixels = width;
	int RowBytes = (width * 3 + 3) & (~3);
	ip.Hbytes = RowBytes;

	printf("\n   Input BMP File name: %20s  (%u x %u)", filename, ip.Hpixels,
			ip.Vpixels);

	pel **TheImage = (pel **) malloc(height * sizeof(pel*));
	for (unsigned int i = 0; i < height; i++)
		TheImage[i] = (pel *) malloc(RowBytes * sizeof(pel));

	for (unsigned int i = 0; i < height; i++)
		fread(TheImage[i], sizeof(unsigned char), RowBytes, f);

	fclose(f);
	return TheImage;  // remember to free() it in caller!
}

/*
 * Store a BMP image
 */
void WriteBMP(pel** img, char* filename) {
	FILE* f = fopen(filename, "wb");
	if (f == NULL) {
		printf("\n\nFILE CREATION ERROR: %s\n\n", filename);
		exit(1);
	}

	//write header
	for (unsigned int x = 0; x < 54; x++)
		fputc(ip.HeaderInfo[x], f);

	//write data
	for (unsigned int x = 0; x < ip.Vpixels; x++)
		for (unsigned int y = 0; y < ip.Hbytes; y++) {
			char temp = img[x][y];
			fputc(temp, f);
		}

	printf("\n  Output BMP File name: %20s  (%u x %u)", filename, ip.Hpixels,
			ip.Vpixels);

	fclose(f);
}
