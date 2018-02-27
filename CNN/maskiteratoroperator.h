#ifndef MASKITERATOROPERATOR_H
#define MASKITERATOROPERATOR_H

#include "crabtensor.h"

/**
 * @brief The MaskIteratorOperator class uses template method pattern
 */

class MaskIteratorOperator{
public:
    MaskIteratorOperator();
    MaskIteratorOperator(CRAbTensor * input, CRAbTensor * mask);

    void iterate();
    virtual void action(){}
    virtual void finalAction(){}

    // setters & getters --------------------------------------------------------------------------
    CRAbTensor getOutput();
    virtual void Delete(){}
protected:
    CRAbTensor *inputOperand,  *maskOperand, output;
    int *currentInput, *currentOutput, *currentMask, *currentConvolution;
    int currentInputIndex, currentOutputIndex, currentMaskIndex, currentConvolutionIndex;
    int * stride;
};

#endif // MASKITERATOROPERATOR_H
