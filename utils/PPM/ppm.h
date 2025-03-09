#ifndef PPM_H_
#define PPM_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define color unsigned char

typedef struct {
    color r;  
    color g;  
    color b;  
} pel;

typedef struct {
    int width, height, maxval;
    color *image;
} PPM;

PPM *ppm_load(const char *filename); // Set pel (pixel element) in ppm image.
void ppm_write(PPM *ppm, const char *filename); // Get pel (pixel element) from ppm image.
pel ppm_get(PPM *ppm, int x, int y);   // Load ppm image from file.
void ppm_set(PPM *ppm, int x, int y, pel c); // Create a copy of ppm image.
PPM *ppm_copy(PPM *ppm); // Create a new ppm image (width x height) with all pixels set to c.
PPM *ppm_make(int width, int height, pel c); // Create a new ppm image (width x height) with random pixel values.
PPM *ppm_rand(int width, int height); // Flip horizontally in place by swapping columns.
void ppm_flipH(PPM *ppm); // Flip vertically in place by swapping columns.
void ppm_flipV(PPM *ppm); // Flip horizontally in place by swapping rows.
void ppm_flipH_row(PPM *ppm); // Flip vertically in place by swapping rows.
void ppm_flipV_col(PPM *ppm); // Compare two ppm images.
int ppm_equal(PPM *ppm1, PPM *ppm2); // Free ppm image.

#endif