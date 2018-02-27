#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "convolver.h"

class Convolution : public Convolver {
public:
    Convolution(int inputOrder, int * inputDimensions, int * maskDimensions, int * padding, int channels, int batchSize);
    void sendError();
    long getSizeBytes();
    void receiveInput(CRAbTensor * input);
    int* getInputShape();
private:
    int * padding;
    CRAbTensor * inputNoPad;
};

#endif // CONVOLUTION_H
