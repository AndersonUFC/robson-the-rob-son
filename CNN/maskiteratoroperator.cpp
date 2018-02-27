#include "maskiteratoroperator.h"

// constructors -----------------------------------------------------------------------------------

MaskIteratorOperator::MaskIteratorOperator(){
    inputOperand = NULL;
    maskOperand = NULL;
    currentInput = NULL;
    currentOutput = NULL;
    currentMask = NULL;
    currentConvolution = NULL;
    currentInputIndex = 0;
    currentMaskIndex = 0;
    currentOutputIndex = 0;
    currentConvolutionIndex = 0;
    stride = NULL;
}

/*

    input has the shape form (data_1 , ... , data_n)
    mask has the shape form (channels, data_1, ... , data_n)
    output has the shape form (channels*data_output_1 , ... , data_output_n)

*/
MaskIteratorOperator::MaskIteratorOperator(CRAbTensor * input, CRAbTensor * mask){
    int inputOrder = input->getOrder();
    stride = new int[inputOrder];
    for(int i = 0 ; i < inputOrder ; i++)
        stride[i] = 1;

    inputOperand = input;
    maskOperand = mask;

    currentInputIndex = 0;
    currentOutputIndex = 0;
    currentMaskIndex = 0;
    currentConvolutionIndex = 0;

    // initialize array indexes
    currentInput = new int[inputOrder];
    currentMask = new int[maskOperand->getOrder()];
    currentConvolution = new int[inputOrder];
}

CRAbTensor MaskIteratorOperator::getOutput(){
    return output.copy();
}

void MaskIteratorOperator::iterate(){
    // get input info
    int inputOrder = inputOperand->getOrder();
    int * inputDimensions = inputOperand->getDimensions();
    int inputSize = inputOperand->getSize();

    // get mask info
    int maskOrder = maskOperand->getOrder();
    int * maskDimensions = maskOperand->getDimensions();
    int maskSize = maskOperand->getSize();
    int maskLastDimension = maskDimensions[maskOrder-1];

    // initialize input index
    for(int i = 0 ; i < inputOrder ; i++)
        currentInput[i] = 0;
    int lastDimension = inputDimensions[inputOrder-1];

    currentInputIndex = -1;
    currentOutputIndex = -1;
    currentConvolutionIndex = -1;
    currentMaskIndex = -1;

    // iterate over input
    while(currentInputIndex + 1 < inputSize){
        // iterate over last dimension
        for(int i1 = 0 ; i1 < lastDimension ; i1 += stride[inputOrder-1]){

            currentInput[inputOrder-1] = i1;
            currentInputIndex = inputOperand->fromDimensionToInt(currentInput);

            // mask iteration ==============================================================
            for(int i2 = 0 ; i2 < maskOrder ; i2++)
                currentMask[i2] = 0;
            currentMaskIndex = -1;

            // iterate over mask
            while(currentMaskIndex + 1 < maskSize){
                for(int i2 = 0 ; i2 < maskLastDimension ; i2++){
                    currentMask[maskOrder-1] = i2;
                    currentMaskIndex = maskOperand->fromDimensionToInt(currentMask);

                    // virtual action
                    action();
                }

                for(int i2 = maskOrder-2 ; i2 >= 0 ; i2--){
                    if(currentMask[i2] + 1 < maskDimensions[i2]){
                        currentMask[i2]++;
                        break;
                    } else {
                        if(i2 > 0){
                            currentMask[i2] = 0;
                            currentMask[i2-1]++;

                            if(currentMask[i2-1] < maskDimensions[i2-1])
                                break;
                        }
                    }
                }
            }
            // mask iteration ==============================================================
        }

        // iterate most significant indices
        for(int i1 = inputOrder-1 ; i1 >= 0 ; i1--){
            if(currentInput[i1] + stride[i1] < inputDimensions[i1]){
                currentInput[i1] += stride[i1];
                break;
            } else {
                if(i1 > 0){
                    currentInput[i1] = 0;
                    currentInput[i1-1] += stride[i1-1];

                    if(currentInput[i1-1] < inputDimensions[i1-1])
                        break;
                }
            }
        }
    }

    // last comands
    finalAction();
}
