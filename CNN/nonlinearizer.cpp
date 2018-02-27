#include "nonlinearizer.h"

// constructor ------------------------------------------------------------------------------------

Nonlinearizer::Nonlinearizer(int order, int * dimensions, int batchSize){
    this->batchSize = batchSize;

    input = new CRAbTensor[batchSize];
    output = new CRAbTensor[batchSize];
    error = new CRAbTensor[batchSize];

    for(int i = 0 ; i < batchSize ; i++){
        input[i] = CRAbTensor(order, dimensions, CT_NULL);
        output[i] = CRAbTensor(order, dimensions, CT_NULL);
    }

    type = CNN_NONLINEARIZER;
    weightError = NULL;
    backwardError = NULL;
    nextLayer = NULL;
    previousLayer = NULL;
}


// virtual methods --------------------------------------------------------------------------------

void Nonlinearizer::deleteLayer(){
    for(int i = 0 ; i < batchSize ; i++){
        input[i].deleteTensor();
        output[i].deleteTensor();
    }

    delete [] input;
    delete [] output;
}

void Nonlinearizer::receiveInput(CRAbTensor * input){
    for(int i = 0 ; i < batchSize ; i++){
        this->input[i].putdata(input[i].getdata());
    }
}

long Nonlinearizer::getSizeBytes(){
    return 2*batchSize*(input[0].getSizeBytes()) + 2*sizeof(int);
}
