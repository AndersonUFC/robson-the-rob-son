#include "poolingoperation.h"

PoolingOperation::PoolingOperation():
    MaskIteratorOperator()
{

}

PoolingOperation::PoolingOperation(CRAbTensor * input, CRAbTensor * mask, int * stride):
    MaskIteratorOperator(input,mask)
{

    int inputOrder = input->getOrder();
    int * inputDimensions = input->getDimensions();

    int * outputDimensions = new int[inputOrder];
    for(int i = 0 ; i < inputOrder ; i++)
        outputDimensions[i] = (inputDimensions[i]-1)/stride[i] + 1;

    output = CRAbTensor(inputOrder, outputDimensions, CT_MIN_NEG);
    currentOutput = new int[inputOrder];
    indicatorMatrix = CRAbTensor(inputOrder, outputDimensions, CT_NULL);
    //this->stride = new int[inputOrder];
    for(int i = 0 ; i < inputOrder ; i++)
        this->stride[i] = stride[i];

    delete [] outputDimensions;
}

void PoolingOperation::action(){
    bool outOfRange = false;
    int inputOrder = inputOperand->getOrder();
    int * inputDimensions = inputOperand->getDimensions();

    for(int i = 0 ; i < inputOrder ; i++)
        if(currentInput[i] + currentMask[i] >= inputDimensions[i])
            outOfRange = true;

    if(!outOfRange){

        for(int i = 0 ; i < inputOrder ; i++){
            currentConvolution[i] = currentInput[i] + currentMask[i];
            currentOutput[i] = currentInput[i]/stride[i];
        }

        currentConvolutionIndex = inputOperand->fromDimensionToInt(currentConvolution);
        currentOutputIndex = output.fromDimensionToInt(currentOutput);

        if(output[currentOutputIndex] < (*inputOperand)[currentConvolutionIndex]){
            output[currentOutputIndex] = (*inputOperand)[currentConvolutionIndex];
            indicatorMatrix[currentOutputIndex] = currentConvolutionIndex;
        }
    }
}

void PoolingOperation::finalAction(){
}

CRAbTensor PoolingOperation::getIndicatorMatrix(){
    return indicatorMatrix.copy();
}

void PoolingOperation::Delete(){
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
    indicatorMatrix.deleteTensor();
    output.deleteTensor();
}
