#ifndef CONVOLVER_H
#define CONVOLVER_H

#include "layer.h"
#include "convolutionoperation.h"
#include "convolutionbackpropagationoperation.h"

class Convolver : public Layer{
public:
    Convolver(int numberOfChannels);

    void run(int currentBatch);
    void update(float learningRate);
    void deleteLayer();
    CRAbTensor getBias(){return bias;}
protected:
    int numberChannels;
    CRAbTensor bias;
    ConvolutionOperation iterator;
    ConvolutionBackPropagationOperation errorIterator;

    void convolve(int inputBatch);
    void updateBias(float learningRate);
};

#endif // CONVOLVER_H
