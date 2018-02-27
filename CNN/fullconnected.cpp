#include "fullconnected.h"

FullConnected::FullConnected(int inputOrder, int * inputDimensions, int hiddenUnits, int batchSize):
    Convolution(inputOrder, inputDimensions, inputDimensions, NULL, hiddenUnits, batchSize)
{
    type = CNN_FULL_CONNECTED;
    iterator.setMaskDivisor(1);
}
