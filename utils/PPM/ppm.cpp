#include <stdio.h>
#include <stdlib.h>
#include "ppm.h"

/*
 * Set pel (pixel element) in ppm image.
 */
 void ppm_set(PPM* ppm, int x, int y, pel c) {
    if (x < 0 || x >= ppm->width || y < 0 || y > ppm->height) {
        printf("Index out of bounds in ppm_set(%d, %d)\n", x, y);
        exit(EXIT_FAILURE);
    }
    int i = x + y*ppm->width;
    ppm->image[3*i] = c.r;
    ppm->image[3*i + 1] = c.g;
    ppm->image[3*i + 2] = c.b;  
}

/*
 * Get pel (pixel element) from ppm image.
 */
 pel ppm_get(PPM* ppm, int x, int y) {     
    if (x < 0 || x >= ppm->width || y < 0 || y > ppm->height) {
        printf("Index out of bounds in ppm_get(%d, %d)\n", x, y);
        exit(EXIT_FAILURE);
    }
    
    pel p;
    int i = x + y*ppm->width;
    p.r = ppm->image[3*i];
    p.g = ppm->image[3*i + 1];
    p.b = ppm->image[3*i + 2];
    return p;
}

/*
* Load ppm image from file.
*/
PPM *ppm_load(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Unable to open file %s\n", filename);
        return NULL;
    }
    // header
    char format[2];
    int maxval;
    int width, height;
    fscanf(fp, "%c %c\n", &format[0], &format[1]);
    fscanf(fp, "%d %d\n%d", &width, &height, &maxval);
    if (format[0] != 'P' || format[1] != '6') {
        printf("PPM Format not supported!\n");
        fclose(fp);
        return NULL;
    }
    // read image
    PPM *ppm = (PPM *)malloc(sizeof(PPM));
    ppm->image = (color *)malloc(3 * width * height);
    ppm->width = width;
    ppm->height = height;
    ppm->maxval = maxval;
    fread(ppm->image, 3, width * height, fp);
    fclose(fp);
    return ppm;
}

/*
* Create a new ppm image (width x height) with all pixels set to c.
*/
PPM *ppm_make(int width, int height, pel c) {
    PPM *ppm = (PPM *)malloc(sizeof(PPM));
    ppm->image = (color *)malloc(3 * width * height);
    ppm->width = width;
    ppm->height = height;
    ppm->maxval = 255;
    for (int i = 0; i < width * height; i++) {
        ppm->image[3*i] = c.r;
        ppm->image[3*i + 1] = c.g;
        ppm->image[3*i + 2] = c.b;
    }
    return ppm;
}

/*
* Create a new ppm image (width x height) with random pixel values.
*/
PPM *ppm_rand(int width, int height) {
    PPM *ppm = (PPM *)malloc(sizeof(PPM));
    ppm->image = (color *)malloc(3 * width * height);
    ppm->width = width;
    ppm->height = height;
    ppm->maxval = 255;
    for (int i = 0; i < width * height; i++) {
        ppm->image[3*i] = rand() % 256;
        ppm->image[3*i + 1] = rand() % 256;
        ppm->image[3*i + 2] = rand() % 256;
    }
    return ppm;
}

/*
 * Write ppm image to file path.
 */
 void ppm_write(PPM* ppm, const char* path) {
    FILE* fp;
    int x, y;

    fp = fopen(path, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", ppm->width, ppm->height);
    fprintf(fp, "%d\n", ppm->maxval);
    fwrite(ppm->image, ppm->width*3, ppm->height, fp);
    fclose(fp);
}

/*
 * Create a copy of ppm image.
 */
 PPM *ppm_copy(PPM* ppm) {
    PPM *ppm1 = (PPM *)malloc(sizeof(PPM));
    ppm1->image = (color *)malloc(3 * ppm->width * ppm->height);
    ppm1->width = ppm->width;
    ppm1->height = ppm->height;
    ppm1->maxval = ppm->maxval;
    memcpy(ppm1->image, ppm->image, 3 * ppm->width * ppm->height);
    return ppm1;
}

/*
 * Flip horizontally in place by swapping column elements.
 */   
 void ppm_flipH(PPM* ppm) {
    for (int x = 0; x < ppm->width/2; x++) {
        for (int y = 0; y < ppm->height; y++) {
            pel p1 = ppm_get(ppm, x, y);
            pel p2 = ppm_get(ppm, ppm->width - x - 1, y);
            ppm_set(ppm, x, y, p2);
            ppm_set(ppm, ppm->width - x - 1, y, p1);
        }
    }
}

/*
* Flip vertically in place by swapping rows.
*/
void ppm_flipV_row(PPM *ppm) {
    int row_size = ppm->width * 3;
    color *temp_row = (color *)malloc(row_size);

    for (int j = 0; j < ppm->height/2; j++) {
        color *row1 = ppm->image + j * row_size;
        color *row2 = ppm->image + (ppm->width - 1 - j) * row_size;
        memcpy(temp_row, row1, row_size);
        memcpy(row1, row2, row_size);
        memcpy(row2, temp_row, row_size);
    }
    free(temp_row);
}

/*
 * Flip vertically in place by swapping row elements.
 */
 void ppm_flipV(PPM* ppm) {
    for (int y = 0; y < ppm->height/2; y++) {
        for (int x = 0; x < ppm->width; x++) {
            pel p1 = ppm_get(ppm, x, y);
            pel p2 = ppm_get(ppm, x, ppm->height - y - 1);
            ppm_set(ppm, x, y, p2);
            ppm_set(ppm, x, ppm->height - y - 1, p1);
        }
    }
}

/*
 * Check if two ppm images are equal.
 */
int ppm_equal(PPM* ppm1, PPM* ppm2) {
    if (ppm1->width != ppm2->width || ppm1->height != ppm2->height) {
        return 0;
    }
    for (int i = 0; i < ppm1->width * ppm1->height; i++) {
        if (ppm1->image[3*i] != ppm2->image[3*i] ||
            ppm1->image[3*i + 1] != ppm2->image[3*i + 1] ||
            ppm1->image[3*i + 2] != ppm2->image[3*i + 2]) {
            return 0;
        }
    }
    return 1;
}

/*
* Blurring filter for ppm images
*/     
void ppm_blur(PPM* ppm, PPM *ppm_filtered, int KERNEL_SIZE) {
    for (int x = 0; x < ppm->width; x++) {
        for (int y = 0; y < ppm->height; y++) {
            pel p = ppm_blurKernel(ppm, x, y, ppm->width, ppm->height, KERNEL_SIZE);
            ppm_set(ppm_filtered, x, y, p);
        }
    }
}

/*
* blur kernel
*/
pel ppm_blurKernel(PPM *ppm, int x, int y, int width, int height, int KERNEL_SIZE) {
    float R=0, G=0, B=0;
    int numPixels = 0;
    int RADIUS = KERNEL_SIZE/2;
    for(int r = -RADIUS; r < RADIUS; ++r) {
        for(int c = -RADIUS; c < RADIUS; ++c) {
            int row = y + r;
            int col = x + c;
            if(row > -1 && row < height && col > -1 && col < width) {
                pel p = ppm_get(ppm, col, row);
                R += p.r;
                G += p.g;
                B += p.b;
                numPixels++;
            }
        }
    }
    pel p_fil = {(color)(R/numPixels), (color)(G/numPixels), (color)(B/numPixels)};
    return p_fil;
}   
// Sharpening filters for ppm images   
void ppm_sharpen(PPM* ppm) {
    PPM *ppm1 = ppm_copy(ppm);
    for (int x = 1; x < ppm->width - 1; x++) {
        for (int y = 1; y < ppm->height - 1; y++) {
            pel p1 = ppm_get(ppm1, x - 1, y - 1);
            pel p2 = ppm_get(ppm1, x, y - 1);
            pel p3 = ppm_get(ppm1, x + 1, y - 1);
            pel p4 = ppm_get(ppm1, x - 1, y);
            pel p5 = ppm_get(ppm1, x, y);
            pel p6 = ppm_get(ppm1, x + 1, y);
            pel p7 = ppm_get(ppm1, x - 1, y + 1);
            pel p8 = ppm_get(ppm1, x, y + 1);
            pel p9 = ppm_get(ppm1, x + 1, y + 1);
            pel p;
            p.r = 5 * p5.r - p2.r - p4.r - p6.r - p8.r;
            p.g = 5 * p5.g - p2.g - p4.g - p6.g - p8.g;
            p.b = 5 * p5.b - p2.b - p4.b - p6.b - p8.b;
            ppm_set(ppm, x, y, p);
        }
    }
    free(ppm1->image);
    free(ppm1);
}
