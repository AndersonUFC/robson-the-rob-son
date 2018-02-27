#include "convolution.h"

Convolution::Convolution(int inputOrder, int * inputDimensions, int * maskDimensions, int * padding, int channels, int batchSize):
    Convolver(channels)
{
    inputNoPad = new CRAbTensor[1];
    inputNoPad[0] = CRAbTensor(inputOrder, inputDimensions, CT_NULL);

    // set batchSize
    this->batchSize = batchSize;

    // create padding array
    this->padding = new int[inputOrder];

    if(padding != NULL){
        for(int i = 0 ; i < inputOrder ; i++)
            this->padding[i] = padding[i];
    } else {
        for(int i = 0 ; i < inputOrder ; i++)
            this->padding[i] = 0;
    }

    // create weights tensor
    int * weightDimensions = new int[inputOrder+1];
    weightDimensions[0] = channels;
    for(int i = 1 ; i < inputOrder+1 ; i++)
        weightDimensions[i] = maskDimensions[i-1];

    weights = CRAbTensor(inputOrder+1 , weightDimensions, CT_RANDOM_POS);

    // create input and output array
    int * newInputDimension = new int[inputOrder];
    int * newOutputDimension = new int[inputOrder];

    for(int i = 0 ; i < inputOrder ; i++)
        newInputDimension[i] = inputDimensions[i] + 2*this->padding[i];

    newOutputDimension[0] = (newInputDimension[0] - maskDimensions[0] + 1) * channels;
    for(int i = 1 ; i < inputOrder ; i++)
        newOutputDimension[i] = newInputDimension[i] - maskDimensions[i] + 1;


    input = new CRAbTensor[batchSize];
    output = new CRAbTensor[batchSize];
    for(int i = 0 ; i < batchSize ; i++){
        input[i] = CRAbTensor(inputOrder, newInputDimension, CT_ZEROS);
        output[i] = CRAbTensor(inputOrder, newOutputDimension, CT_NULL);
    }

    backwardError = new CRAbTensor[batchSize];
    weightError = new CRAbTensor[batchSize];

    error = new CRAbTensor[batchSize];

    // set some pointers to NULL

    nextLayer = NULL;
    previousLayer = NULL;
    type = CNN_CONVOLUTION;

    // free temp pointers

    delete [] weightDimensions;
    delete [] newInputDimension;
    delete [] newOutputDimension;

}

void Convolution::sendError(){

    const int uni_batch = 1;
    int order = error[0].getOrder();
    int * oldDimension = new int[order], * newDimension = new int[order+1];
    int * maskDimensions = weights.getDimensions();

    //weight error
    for(int i = 0 ; i < batchSize ; i ++){
        int * err_dim = error[i].getDimensions();

        for(int j = 0 ; j < order ; j++)
            oldDimension[j] = err_dim[j];

        newDimension[0] = maskDimensions[0];
        newDimension[1] = oldDimension[0]/newDimension[0];

        for(int j = 1 ; j < order ; j++)
            newDimension[j+1] = oldDimension[j];
        error[i].reshape(order+1, newDimension);

        iterator.Delete();
        iterator = ConvolutionOperation(&input[i], &error[i], NULL, false);
        iterator.iterate();

        error[i].reshape(order, oldDimension);
        this->weightError[i].deleteTensor();
        this->weightError[i] = iterator.getOutput();
    }

    delete [] oldDimension;
    delete [] newDimension;

    int weOrder = weightError[0].getOrder();
    int * weDim = weightError[0].getDimensions();

    newDimension = new int[weOrder+1];

    newDimension[0] = maskDimensions[0];
    newDimension[1] = weDim[0]/newDimension[0];

    for(int i = 2 ; i < weOrder+1 ; i++)
        newDimension[i] = weDim[i-1];
    // reshape all errors
    for(int i = 0 ; i < batchSize ; i++)
        weightError[i].reshape(weOrder+1, newDimension);

    delete [] newDimension;
    // backward error -----------------------------------------------------
    for(int i = 0 ; i < batchSize ; i++){
        errorIterator.Delete();
        errorIterator = ConvolutionBackPropagationOperation(&error[i], &weights);
        errorIterator.iterate();

        backwardError[i].deleteTensor();
        backwardError[i] = errorIterator.getOutput();
    }


    int beOrder = backwardError[0].getOrder();
    int size = inputNoPad[0].getSize();

    int * inputDimensions = inputNoPad[0].getDimensions();
    int lastDimension = inputDimensions[beOrder-1];

    int * currentInputPad = new int[beOrder];
    int * currentInputNoPad = new int[beOrder];


    // reshape error if have pad
    CRAbTensor reshapedError;
    for(int j = 0 ; j < batchSize ; j++){

        reshapedError = CRAbTensor(beOrder, inputDimensions, CT_NULL);
        int currentInputPadIndex = -1, currentInputNoPadIndex = -1;

        for(int i = 0 ; i < beOrder ; i++){
            currentInputNoPad[i] = 0;
            currentInputPad[i] = padding[i];
        }

        while(currentInputNoPadIndex + 1 < size){
            // iterate over last dimension
            for(int i = 0 ; i < lastDimension ; i++){
                currentInputNoPad[beOrder-1] = i;
                currentInputPad[beOrder-1] = padding[beOrder-1] + i;

                currentInputNoPadIndex = inputNoPad[0].fromDimensionToInt(currentInputNoPad);
                currentInputPadIndex = input[0].fromDimensionToInt(currentInputPad);

                reshapedError[currentInputNoPadIndex] = backwardError[j][currentInputPadIndex];
            }

            // iterate over indexes
            for(int i = beOrder - 2 ; i >= 0 ; i--){
                if(currentInputNoPad[i] + 1 < inputDimensions[i]){
                    currentInputNoPad[i]++;
                    currentInputPad[i]++;
                    break;
                } else {
                    if(i > 0){
                        currentInputNoPad[i] = 0;
                        currentInputPad[i] = padding[i];

                        currentInputNoPad[i-1]++;
                        currentInputPad[i-1]++;

                        if(currentInputNoPad[i] < inputDimensions[i])
                            break;
                    }
                }
            }
        }

        backwardError[j].deleteTensor();
        backwardError[j] = reshapedError;
    }

    delete [] currentInputPad;
    delete [] currentInputNoPad;

    previousLayer->receiveError(backwardError);
    previousLayer->sendError();
}

long Convolution::getSizeBytes(){

}

void Convolution::receiveInput(CRAbTensor * input){
    int inputOrder = this->input[0].getOrder();
    int inputSize = input[0].getSize();
    int * currentInputPad = new int[inputOrder];
    int * currentInputNoPad = new int[inputOrder];
    int * inputDimensions = input[0].getDimensions();

    int lastDimension = inputDimensions[inputOrder-1];

    // for every input example
    for(int i1 = 0 ; i1 < batchSize ; i1++){
        // initialize indexes
        for(int i2 = 0 ; i2 < inputOrder ; i2++){
            currentInputPad[i2] = padding[i2];
            currentInputNoPad[i2] = 0;
        }

        // iterate on indexes
        int currentInputNoPadIndex = -1;
        int currentInputPadIndex = -1;
        while(currentInputNoPadIndex + 1 < inputSize){
            // iterate over last dimension
            for(int i2 = 0 ; i2 < lastDimension ; i2++){
                currentInputNoPad[inputOrder-1] = i2;
                currentInputPad[inputOrder-1] = padding[inputOrder-1] + i2;

                currentInputNoPadIndex = input[0].fromDimensionToInt(currentInputNoPad);
                currentInputPadIndex = this->input[0].fromDimensionToInt(currentInputPad);

                this->input[i1][currentInputPadIndex] = input[i1][currentInputNoPadIndex];
            }

            // update indexes
            for(int i2 = inputOrder-2 ; i2 >= 0 ; i2--){
                if(currentInputNoPad[i2] + 1 < inputDimensions[i2]){
                    currentInputNoPad[i2]++;
                    currentInputPad[i2]++;
                    break;
                } else {
                    if(i2 > 0){
                        currentInputNoPad[i2-1]++;
                        currentInputPad[i2-1]++;

                        currentInputNoPad[i2] = 0;
                        currentInputPad[i2] = padding[i2];

                        if(currentInputNoPad[i2] < inputDimensions[i2])
                            break;
                    }
                }
            }
        }
    }

    delete [] currentInputPad;
    delete [] currentInputNoPad;
}

int* Convolution::getInputShape(){
    return inputNoPad[0].getDimensions();
}
