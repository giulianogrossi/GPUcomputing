struct imgBMP {
	int width;
	int height;
	unsigned char headInfo[54];
	unsigned long int rowByte;
} img;

#define	WIDTHB		img.rowByte
#define	WIDTH		img.width
#define	HEIGHT		img.height
#define	IMAGESIZE	(WIDTHB*HEIGHT)

struct pixel {
	unsigned char R;
	unsigned char G;
	unsigned char B;
};

typedef unsigned long ulong;
typedef unsigned int uint;
typedef unsigned char pel;    // pixel element

pel *ReadBMPlin(char*);         // Load a BMP image
void WriteBMPlin(pel *, char*); // Store a BMP image
