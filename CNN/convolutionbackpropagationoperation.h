#ifndef CONVOLUTIONBACKPROPAGATIONOPERATION_H
#define CONVOLUTIONBACKPROPAGATIONOPERATION_H

#include "maskiteratoroperator.h"

class ConvolutionBackPropagationOperation : public MaskIteratorOperator
{
public:
    ConvolutionBackPropagationOperation(){}
    ConvolutionBackPropagationOperation(CRAbTensor * input, CRAbTensor * mask);
    void action();
    void finalAction();
    void Delete();
private:
    int * inputStride;
    CRAbTensor ocurrence;
};

#endif // CONVOLUTIONBACKPROPAGATIONOPERATION_H
