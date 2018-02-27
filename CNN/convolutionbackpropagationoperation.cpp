#include "convolutionbackpropagationoperation.h"

ConvolutionBackPropagationOperation::ConvolutionBackPropagationOperation(CRAbTensor * input, CRAbTensor * mask):
    MaskIteratorOperator(input, mask)
{

    int inputOrder = input->getOrder();
    int * inputDimensions = input->getDimensions();
    int * maskDimensions = mask->getDimensions();
    int * outputDimension = new int[inputOrder];

    outputDimension[0] = inputDimensions[0]/maskDimensions[0] + maskDimensions[1] - 1;
    for(int i = 1 ; i < inputOrder ; i++)
        outputDimension[i] = inputDimensions[i] + maskDimensions[i+1] - 1;

    output = CRAbTensor(inputOrder, outputDimension, CT_ZEROS);
    ocurrence = CRAbTensor(inputOrder, outputDimension, CT_ZEROS);
    currentOutput = new int[inputOrder];

    // reshape input
    int * newInputDimension = new int[inputOrder+1];
    newInputDimension[0] = maskDimensions[0];
    newInputDimension[1] = inputDimensions[0]/maskDimensions[0];

    for(int i = 2 ; i < inputOrder+1 ; i++)
        newInputDimension[i] = inputDimensions[i-1];

    input->reshape(inputOrder+1, newInputDimension);



    // reshape stride
    inputOrder = input->getOrder();
    delete [] stride;
    stride = new int[inputOrder];
    for(int i = 0 ; i < inputOrder ; i++)
        stride[i] = 1;

    if(currentInput != NULL){
	delete [] currentInput;
	currentInput = new int[inputOrder];
    }

    delete [] outputDimension;
    delete [] newInputDimension;
}

void ConvolutionBackPropagationOperation::action(){
    if(currentInput[0] == currentMask[0]){
        int outputOrder = output.getOrder();

        for(int i = 0 ; i < outputOrder ; i++)
            currentOutput[i] = currentInput[i+1] + currentMask[i+1];

        currentOutputIndex = output.fromDimensionToInt(currentOutput);
        output[currentOutputIndex] += (*inputOperand)[currentInputIndex]*(*maskOperand)[currentMaskIndex];
        ocurrence[currentOutputIndex]++;
    }
}

void ConvolutionBackPropagationOperation::finalAction(){
    // reshape error input
    int inputOrder = inputOperand->getOrder();
    int * inputDimensions = inputOperand->getDimensions();
    int * newInputDimensions = new int[inputOrder-1];

    newInputDimensions[0] = inputDimensions[0]*inputDimensions[1];
    for(int i = 1 ; i < inputOrder-1 ; i++)
        newInputDimensions[i] = inputDimensions[i+1];

    inputOperand->reshape(inputOrder-1, newInputDimensions);

    delete [] newInputDimensions;

    // output
    int size = output.getSize();
    for(int i = 0 ; i < size ; i++)
        if(ocurrence[i] != 0)
            output[i] /= ocurrence[i];
}

void ConvolutionBackPropagationOperation::Delete(){
    if(this->currentConvolution != NULL){
        delete [] this->currentConvolution;
        this->currentConvolution = NULL;
    }
    if(this->currentInput != NULL){
        delete [] this->currentInput;
        this->currentInput = NULL;
    }
    if(this->currentMask != NULL){
        delete [] this->currentMask;
        this->currentMask = NULL;
    }
    if(this->currentOutput != NULL){
        delete [] this->currentOutput;
        this->currentOutput = NULL;
    }
    if(this->stride != NULL){
        delete [] this->stride;
        this->stride = NULL;
    }
    output.deleteTensor();
    ocurrence.deleteTensor();
}
