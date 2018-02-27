#include "input.h"

// constructor ====================================================================================

Input::Input(int order, int * dimensions, int batchSize)
{
    // create batch
    this->batchSize = batchSize;
    output = new CRAbTensor[batchSize];
    for(int i = 0 ; i < batchSize ; i++)
        output[i] = CRAbTensor(order, dimensions, CT_NULL);

    // null pointers
    input = NULL;
    error = NULL;
    weightError = NULL;
    backwardError = NULL;
    nextLayer = NULL;
    previousLayer = NULL;

    type = CNN_INPUT;
}

// virtual methods ================================================================================

void Input::deleteLayer(){

    for(int i = 0 ; i < batchSize ; i++)
        output[i].deleteTensor();
    delete [] output;

    nextLayer = NULL;
    previousLayer = NULL;
}

void Input::receiveInput(CRAbTensor * input){
    for(int i = 0 ; i < batchSize ; i++)
        output[i].putdata(input[i].getdata());
}


long Input::getSizeBytes(){
    return batchSize * output[0].getSizeBytes() + sizeof(int);
}
