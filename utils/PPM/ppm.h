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

PPM *ppm_load(const char *filename);
void ppm_write(PPM *ppm, const char *filename);
pel ppm_get(PPM *ppm, int x, int y);   
void ppm_set(PPM *ppm, int x, int y, pel c);
PPM *ppm_copy(PPM *ppm);
PPM *ppm_make(int width, int height, pel c);
PPM *ppm_rand(int width, int height);
void ppm_flipH(PPM *ppm);
void ppm_flipV(PPM *ppm);
void ppm_flipH_row(PPM *ppm);
void ppm_flipV_col(PPM *ppm);
int ppm_equal(PPM *ppm1, PPM *ppm2);


#endif