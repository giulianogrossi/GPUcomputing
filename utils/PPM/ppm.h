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

// Load ppm image from file
PPM *ppm_load(const char *filename); 
// Write ppm image to file
void ppm_write(PPM *ppm, const char *filename); 
// Get pel (pixel element) from ppm image
pel ppm_get(PPM *ppm, int x, int y);   
// Set pel (pixel element) in ppm image
void ppm_set(PPM *ppm, int x, int y, pel c); 
// Create a copy of ppm image
PPM *ppm_copy(PPM *ppm); 
// Create a new ppm image (width x height) with all pixels set to c
PPM *ppm_make(int width, int height, pel c); 
// Create a new ppm image (width x height) with random pixel values.
PPM *ppm_rand(int width, int height); 
// Flip vertically in place by swapping row elements.
void ppm_flipH(PPM *ppm); 
 // Flip horizontally in place by swapping column elements.
void ppm_flipV(PPM *ppm);
// Flip vertically in place by swapping rows
void ppm_flipV_row(PPM *ppm); 
// Compare two ppm images for equality
int ppm_equal(PPM *ppm1, PPM *ppm2); 
// Blurring filter for ppm images
void ppm_blur(PPM *ppm, PPM *ppm_filtered, int KERNEL_SIZE);
// blur kernel
pel ppm_blurKernel(PPM *ppm, int x, int y, int width, int height, int KERNEL_SIZE);


#endif