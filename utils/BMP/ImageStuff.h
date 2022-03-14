struct ImgProp {
	int Hpixels;
	int Vpixels;
	unsigned char HeaderInfo[54];
	unsigned long int Hbytes;
};

struct Pixel {
	unsigned char R;
	unsigned char G;
	unsigned char B;
};

typedef unsigned char pel;    // pixel element

pel** ReadBMP(char*);         // Load a BMP image
void WriteBMP(pel**, char*);  // Store a BMP image

extern struct ImgProp ip;
