#include "convolutionoperation.h"

ConvolutionOperation::ConvolutionOperation():
    MaskIteratorOperator()
{
    bias_turn = NULL;
}

ConvolutionOperation::ConvolutionOperation(CRAbTensor * input, CRAbTensor * mask, CRAbTensor * bias_unit, bool is_convolution_run):
    MaskIteratorOperator(input, mask)
{

    // create output tensor
    int inputOrder = input->getOrder();
    int * inputDimension = input->getDimensions();
    int * maskDimension = mask->getDimensions();
    int * outputDimension = new int[inputOrder+1];

    //mask_divisor = 1;
    mask_divisor = mask->getAccumulator()[0];

    outputDimension[0] = maskDimension[0];
    for(int i = 0 ; i < inputOrder ; i++)
        outputDimension[i+1] =(inputDimension[i] - maskDimension[i+1]) + 1;
    output = CRAbTensor(inputOrder+1, outputDimension, CT_ZEROS);
    currentOutput = new int[inputOrder+1];


    this->bias_turn = NULL;
    this->calculate_bias = false;

    // set bias
    if(is_convolution_run){


        this->calculate_bias = is_convolution_run;
        bias = bias_unit;
        int channels = mask->getDimensions()[0];
        this->bias_turn = new bool[channels];

        mask_divisor += bias->getSize();

        for(int i = 0 ; i < channels ; i++)
	    this->bias_turn[i] = false;
    }

    delete [] outputDimension;
}

void ConvolutionOperation::action(){

    int * inputDimensions = inputOperand->getDimensions();
    int * maskDimensions = maskOperand->getDimensions();

    int inputOrder = inputOperand->getOrder();
    bool outOfRange = false;

    for(int i = 0 ; i < inputOrder ; i++){
        if(currentInput[i] + maskDimensions[i+1] - 1 >= inputDimensions[i])
            outOfRange = true;
    }

    if(! outOfRange){
        // set convolution area
        for(int i = 0 ; i < inputOrder ; i++)
            currentConvolution[i] = currentInput[i] + currentMask[i+1];

        // set output index
        currentOutput[0] = currentMask[0];
        for(int i = 0 ; i < inputOrder ; i++)
            currentOutput[i+1] = currentInput[i];

        currentConvolutionIndex = inputOperand->fromDimensionToInt(currentConvolution);
        currentOutputIndex = output.fromDimensionToInt(currentOutput);

        if(calculate_bias)
            output[currentOutputIndex] += ((*inputOperand)[currentConvolutionIndex] * (*maskOperand)[currentMaskIndex] + ((*bias)[currentOutput[0]])/mask_divisor)/mask_divisor;
        else
            output[currentOutputIndex] += ((*inputOperand)[currentConvolutionIndex] * (*maskOperand)[currentMaskIndex])/mask_divisor;


        /*
        // calculate bias value
        if(calculate_bias && !bias_turn[currentOutput[0]]){
            bias_turn[currentOutput[0]] = true;
            output[currentOutputIndex] += (*bias)[currentOutput[0]]/mask_divisor;
        }
        */
    }


}

void ConvolutionOperation::finalAction(){
    // reshape operation
    int newOrder = output.getOrder()-1;
    int * nod = new int[newOrder];
    int * od = output.getDimensions();

    nod[0] = od[0]*od[1];
    for(int i = 1 ; i < newOrder ; i++)
        nod[i] = od[i+1];

    output.reshape(newOrder, nod);

    delete [] nod;
}


void ConvolutionOperation::Delete(){
    if(this->bias_turn != NULL){
        delete [] this->bias_turn;
        this->bias_turn = NULL;
    }
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
}

void ConvolutionOperation::setMaskDivisor(float md){
    mask_divisor = md;
}
