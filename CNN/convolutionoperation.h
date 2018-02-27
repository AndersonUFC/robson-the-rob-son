#ifndef CONVOLUTIONOPERATION_H
#define CONVOLUTIONOPERATION_H

#include "maskiteratoroperator.h"


class ConvolutionOperation : public MaskIteratorOperator
{
public:
    ConvolutionOperation();
    ConvolutionOperation(CRAbTensor * input, CRAbTensor * mask, CRAbTensor * bias_unit, bool calculate_bias);
    void action();
    void finalAction();
    void Delete();

    void setMaskDivisor(float md);
private:
    float mask_divisor;
    CRAbTensor * bias;
    bool * bias_turn;
    bool calculate_bias;
};

#endif // CONVOLUTIONOPERATION_H
