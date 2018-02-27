#include "relu.h"

// constructor ------------------------------------------------------------------------------------

ReLU::ReLU(int order, int * dimensions, int batchSize):
    Nonlinearizer(order, dimensions, batchSize)
{
    type = CNN_RELU;

    backwardError = new CRAbTensor[batchSize];

    for(int i = 0 ; i < batchSize ; i++)
        backwardError[i] = CRAbTensor(order, dimensions, CT_NULL);
}

// virtual methods --------------------------------------------------------------------------------

void ReLU::run(int currentBatch){
    int size = input[currentBatch].getSize();

    for(int i = 0 ; i < size ; i++){
        if(input[currentBatch][i] > 0){
            output[currentBatch][i] = input[currentBatch][i];
        } else{
            output[currentBatch][i] = 0;
        }
    }
}

void ReLU::sendError(){

    int size = output[0].getSize();
    const int uni_batch = 1;
    for(int i = 0 ; i < batchSize ; i++){
        for(int j = 0 ; j < size ; j++){
            if(input[i][j] > 0){
                backwardError[i][j] = error[i][j];
            } else {
                backwardError[i][j] = 0;
            }
        }
    }


    previousLayer->receiveError(backwardError);
    previousLayer->sendError();
}
