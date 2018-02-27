#ifndef POOLINGOPERATION_H
#define POOLINGOPERATION_H

#include "maskiteratoroperator.h"

class PoolingOperation : public MaskIteratorOperator
{
public:
    PoolingOperation();
    PoolingOperation(CRAbTensor * input, CRAbTensor * mask, int * stride);

    void action();
    void finalAction();
    void Delete();

    CRAbTensor getIndicatorMatrix();
private:
    CRAbTensor indicatorMatrix;
};

#endif // POOLINGOPERATION_H
