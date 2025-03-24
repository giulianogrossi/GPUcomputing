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
    fwrite(ppm->image, 1, 3*ppm->width*ppm->height, fp);
	fprintf(fp, "\n");
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
* Gaussian mask
*/
float *gaussMask(int MASK_SIZE, float SIGMA) {
    float *mask = (float *)malloc(MASK_SIZE * MASK_SIZE * sizeof(float));
    float sum = 0.0;
    int RADIUS = MASK_SIZE / 2;
    float sigma2 = 2.0 * SIGMA * SIGMA;
    
    for (int i = 0; i < MASK_SIZE; i++) {
        for (int j = 0; j < MASK_SIZE; j++) {
            int x = i - RADIUS;
            int y = j - RADIUS;
            mask[i*MASK_SIZE+j] = exp(-(x * x + y * y) / sigma2) / (M_PI * sigma2);
            sum += mask[i*MASK_SIZE+j];
        }
    }
    
    for (int i = 0; i < MASK_SIZE; i++) {
        for (int j = 0; j < MASK_SIZE; j++) {
            mask[i*MASK_SIZE+j] /= sum;
        }
    }
    return mask;
}

/*
* Gaussian filter for ppm images
*/     
void ppm_gaussFilter(PPM* ppm, PPM *ppm_filtered, int MASK_SIZE, float SIGMA) {
    float *mask = gaussMask(MASK_SIZE, SIGMA);
    for (int x = 0; x < ppm->width; x++) {
        for (int y = 0; y < ppm->height; y++) {
            pel p = ppm_gaussKernel(ppm, x, y, ppm->width, ppm->height, MASK_SIZE, mask);
            ppm_set(ppm_filtered, x, y, p);
        }
    }
}

/*
* Gaussian filter
*/
pel ppm_gaussKernel(PPM *ppm, int x, int y, int width, int height, int MASK_SIZE, float *mask) {
    float R=0, G=0, B=0;
    int RADIUS = MASK_SIZE/2;
    for (int r = 0; r < MASK_SIZE; ++r) {
        for (int c = 0; c < MASK_SIZE; ++c) {
            int row = y + r - RADIUS;
            int col = x + c - RADIUS;
            if (row > -1 && row < height && col > -1 && col < width) {
                float m = mask[r * MASK_SIZE + c];
                pel p = ppm_get(ppm, col, row);
                R += p.r * m;
                G += p.g * m;
                B += p.b * m;
            }
        }
    }
    return {(color)R, (color)G, (color)B};
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
* Blur kernel 
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

/* 
* RGB histogram of the PPM image
*/
int *ppm_histogram(PPM *ppm) {
    int *histogram = (int *)malloc(3* 256 * sizeof(int));
    
    // initialize histogram
    for (int x = 0; x < 3 * 256; x++) 
        histogram[x] = 0;

    // count the number of pixels for each color
    for (int x = 0; x < ppm->width * ppm->height; x++) {
        histogram[ppm->image[3*x]]++;
        histogram[ppm->image[3*x+1] + 256]++;
        histogram[ppm->image[3*x+2] + 512]++;
    }
    return histogram;
}

/*
 *  Create and save a histogram as a PPM image
*/
void ppm_save_histogram(int *histogram, const char *filename) {
    // size of the image
    int histSize = 3*256; // R,G,B * 256
    int WIDTH = 3*256*3;  // 3*256 bars, each 3 pixels wide
    int HEIGHT = 500;     //  image height
    int stride = 0;

    // find max of the histogram    
    float max_h = 0.0f;
    for (int i = 0; i < histSize; i++) 
        if (histogram[i] > max_h) 
            max_h = histogram[i];

    // scale histogram to fit in the image
    int *histogram_scaled = (int *)malloc(histSize * sizeof(int));
    for (int i = 0; i < histSize; i++) 
        histogram_scaled[i] = (int)((float)histogram[i] / max_h * HEIGHT);
    
    // Create a new PPM image: canvas for the histogram 
    PPM *ppm_h = ppm_make(WIDTH, HEIGHT, (pel){150, 150, 150});

    // Draw histogram bars for R
    for (int i = 0; i < 256; i++) {
        // make red bars
        int barHeight = histogram_scaled[i];
        for (int y = HEIGHT; y >=  HEIGHT-barHeight; y--) {
            ppm_set(ppm_h, 3*i+1, y, (pel){255, 0, 0});
            ppm_set(ppm_h, 3*i+2, y, (pel){255, 0, 0});
        }   
        // make green bars
        barHeight = histogram_scaled[i+256];
        for (int y = HEIGHT; y >=  HEIGHT-barHeight; y--) {
            int stride = 3*256;
            ppm_set(ppm_h, 3*i+1+stride, y, (pel){0, 255, 0});
            ppm_set(ppm_h, 3*i+2+stride, y, (pel){0, 255, 0});
        }
        // make blue bars
        barHeight = histogram_scaled[i+512];
        for (int y = HEIGHT; y >=  HEIGHT-barHeight; y--) { 
            stride = 6*256;
            ppm_set(ppm_h, 3*i+1+stride, y, (pel){0, 0, 255});
            ppm_set(ppm_h, 3*i+2+stride, y, (pel){0, 0, 255});
        }
    }

    // Save the histogram image
    ppm_write(ppm_h, filename);
    printf("Histogram saved as %s\n", filename);
}


/* 
* Color frequencies in the PPM image
*/
int ppm_freq_color(PPM *ppm, pel c) {    
    int count = 0;

    // count the number of pixels for each color
    for (int x = 0; x < ppm->width * ppm->height; x++) {
        if (ppm->image[3*x] == c.r && ppm->image[3*x+1] == c.g && ppm->image[3*x+2] == c.b) {
            count++;
        }
    }
    return count;
}

/*
* Extract channel: 0 = R, 1 = G, 2 = B
*/
color *ppm_extract_channel(PPM *ppm, int RGB_channel) {
    if (RGB_channel < 0 || RGB_channel > 2) {
        printf("Invalid channel %d\n", RGB_channel);
        exit(EXIT_FAILURE);
    }

    color *ppm_channel = (color *)malloc(ppm->width * ppm->height);
    for (int i = 0; i < ppm->width * ppm->height; i++) {
        ppm_channel[i] = ppm->image[3*i + RGB_channel];
    }
    return ppm_channel;
}

/*
* Combine channels: R, G, B
*/
PPM *ppm_combine_channels(color *R, color *G, color *B, int width, int height) {
    PPM *ppm = (PPM *)malloc(sizeof(PPM));
    ppm->image = (color *)malloc(3 * width * height);
    ppm->width = width;
    ppm->height = height;
    ppm->maxval = 255;
    for (int i = 0; i < width * height; i++) {
        ppm->image[3*i] = R[i];
        ppm->image[3*i + 1] = G[i];
        ppm->image[3*i + 2] = B[i];
    }
    return ppm;
}