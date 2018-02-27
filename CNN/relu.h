#ifndef RELU_H
#define RELU_H

#include "nonlinearizer.h"

class ReLU : public Nonlinearizer
{
public:
    ReLU(int order, int * dimensions, int batchSize);

    void run(int currentBatch);
    void sendError();
};

#endif // RELU_H
