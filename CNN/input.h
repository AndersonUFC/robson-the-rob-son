#ifndef INPUT_H
#define INPUT_H

#include "layer.h"

class Input : public Layer
{
public:
    Input(int order, int * dimensions, int batchSize);
    void deleteLayer();
    void receiveInput(CRAbTensor * input);
    long getSizeBytes();
};

#endif // INPUT_H
