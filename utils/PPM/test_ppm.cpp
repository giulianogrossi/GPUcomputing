
#include "ppm.h"

int main(void) {
    char path[] = "../../images/dog.ppm";
    PPM *img = ppm_load(path);
    printf("PPM image size (w x h): %d x %d\n", img->width, img->height);
        
    // write some values
    for (int i = 0; i < 10; i++) {
        int x = rand() % img->width;
        int y = rand() % img->height;
        pel p = ppm_get(img, x, y);
        printf("image[%d,%d]=(%d %d %d)\n",x,y, p.r, p.g, p.b);
    }
    ppm_write(img, "../../images/output.ppm");

    // create a random image
    int width = 20;
    int height = 10;
    PPM *img1 = ppm_rand(width, height);
    ppm_write(img1, "../../images/test.ppm");

    // flip horizontally
    PPM *img2 = ppm_copy(img);
    ppm_flipH(img2);
    ppm_write(img2, "../../images/output_flippedH.ppm");
    ppm_flipH(img2);
    printf("PPM images are %s\n", ppm_equal(img, img2) ? "equal" : "not equal");
    ppm_write(img2, "../../images/output_flippedH2times.ppm");
    
    // flip vertically  
    PPM *img3 = ppm_copy(img);
    ppm_flipV(img3);
    ppm_write(img3, "../../images/output_flippedV.ppm");

    // blur the image
    PPM *img4 = ppm_make(img->width, img->height, (pel) {0,0,0});
    int MASK_SIZE = 21;
    ppm_blur(img, img4, MASK_SIZE);
    ppm_write(img4, "../../images/output_blurred.ppm");

    // apply gaussian filter to the image
    PPM *img5 = ppm_make(img->width, img->height, (pel) {0,0,0});
    MASK_SIZE = 21;
    float SIGMA = 10.0;
    ppm_gaussFilter(img, img5, MASK_SIZE, SIGMA);
    ppm_write(img5, "../../images/output_gaussian.ppm");

    // histogram
    int *histogram = ppm_histogram(img);
    ppm_save_histogram(histogram, "../../images/histogram.ppm"); 
    
    // color frequency
    pel c = {0, 0, 0};
    int freq = ppm_freq_color(img, c);
    printf("Color frequency of (%u %u %u) is %d\n", c.r, c.g, c.b, freq);

    return 0;
}