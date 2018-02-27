#ifndef NONLINEARIZER_H
#define NONLINEARIZER_H


#include "layer.h"

class Nonlinearizer : public Layer
{
public:
    Nonlinearizer(int order, int * dimensions, int batchSize);

    void deleteLayer();
    void receiveInput(CRAbTensor * input);
    long getSizeBytes();
};

#endif // NONLINEARIZER_H
