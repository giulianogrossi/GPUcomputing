#pragma once
#include "matrix.h"

float computeAccuracy(const Matrix& predictions, const Matrix& targets);
float BCE(Matrix predictions, Matrix target);
Matrix dBCE(Matrix predictions, Matrix target, Matrix dY);