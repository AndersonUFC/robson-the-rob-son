#ifndef POOLING_H
#define POOLING_H

#include "layer.h"
#include "poolingoperation.h"

class Pooling : public Layer
{
public:
    Pooling(int inputOrder, int * inputDimensions, int * maskDimensions, int * stride, int batchSize);
    void run(int currentBatch);
    void receiveInput(CRAbTensor * input);
    void sendError();
private:
    int * stride;
    PoolingOperation iterator;
    CRAbTensor* indicatorMatrix;
};

#endif // POOLING_H

/*
    virtual void sendError(){}
    virtual void deleteLayer(){}
    virtual long getSizeBytes(){}
 */
