#include "mqdb.h"

/*
 * product between mqdb matrices restricted to blocks
 */
void mqdbProd(mqdb A, mqdb B, mqdb C) {
	uint n = 0;
	for (uint i = 0; i < A.nBlocks; i++)
		n += A.blkSize[i];                    // mat dim
	int k = A.nBlocks;                      // num blks
	int dl = 0;                             // blk left bound
	int dr = 0;                             // blk left bound
	for (uint i = 0; i < k; i++) {          // loop on blks
		dr += A.blkSize[i];                   // blk right bound
		for (uint r = dl; r < dr; r++) {      // scan block rows
			for (uint c = dl; c < dr; c++) {    // scan block cols
				float s = 0;
				for (uint l = dl; l < dr; l++)
					s += A.elem[r*n + l] * B.elem[c + l * n];
				C.elem[r*n + c] = s;
			}
		}
		dl = dr;
	}
}