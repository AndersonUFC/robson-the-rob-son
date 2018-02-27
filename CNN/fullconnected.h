#ifndef FULLCONNECTED_H
#define FULLCONNECTED_H

#include "convolution.h"

class FullConnected : public Convolution
{
public:
    FullConnected(int inputOrder, int * inputDimensions, int hiddenUnits, int batchSize);
};

#endif // FULLCONNECTED_H
