#include "pooling.h"

Pooling::Pooling(int inputOrder, int * inputDimensions, int * maskDimensions, int * stride, int batchSize)
{
    this->batchSize = batchSize;

    this->stride = new int[inputOrder];
    for(int i = 0 ; i < inputOrder ; i++)
        this->stride[i] = stride[i];

    int * outputDimensions = new int[inputOrder];
    for(int i = 0 ; i < inputOrder ; i++)
        outputDimensions[i] = (inputDimensions[i]-1)/stride[i] + 1;

    this->type = CNN_POOLING;

    // input and output
    input = new CRAbTensor[batchSize];
    output = new CRAbTensor[batchSize];
    for(int i = 0 ; i < batchSize ; i++){
        input[i] = CRAbTensor(inputOrder, inputDimensions, CT_NULL);
        output[i] = CRAbTensor(inputOrder, outputDimensions, CT_NULL);
    }

    // weights
    weights = CRAbTensor(inputOrder, maskDimensions, CT_NULL);

    // indicator matrix
    indicatorMatrix = new CRAbTensor[batchSize];

    this->backwardError = new CRAbTensor[batchSize];
    // null pointers
    this->error = NULL;
    this->nextLayer = NULL;
    this->previousLayer = NULL;
    this->weightError = NULL;

    // delete temporary pointers
    delete [] outputDimensions;
}

void Pooling::run(int currentBatch){

    iterator.Delete();
    iterator = PoolingOperation(&input[currentBatch], &weights, stride);
    iterator.iterate();

    output[currentBatch].deleteTensor();
    output[currentBatch] = iterator.getOutput();

    indicatorMatrix[currentBatch].deleteTensor();
    indicatorMatrix[currentBatch] = iterator.getIndicatorMatrix();
}

void Pooling::receiveInput(CRAbTensor * input){
    for(int i = 0 ; i < batchSize ; i++)
        this->input[i].putdata(input[i].getdata());
}

void Pooling::sendError(){
    const int uni_batch = 1;
    int inputOrder = input[0].getOrder();
    int * inputDimensions = input[0].getDimensions();
    for(int i = 0 ; i < batchSize ; i++){
        backwardError[i].deleteTensor();
        backwardError[i] = CRAbTensor(inputOrder, inputDimensions, CT_ZEROS);

        int indicatorSize = indicatorMatrix[i].getSize();
        for(int i2 = 0 ; i2 < indicatorSize ; i2++)
            backwardError[i][indicatorMatrix[i][i2]] += error[i][i2];
    }

    // send error backwards
    if(previousLayer != NULL){
        previousLayer->receiveError(backwardError);
        previousLayer->sendError();
    }
}
