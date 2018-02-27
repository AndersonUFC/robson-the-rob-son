#include "output.h"

Output::Output(int inputOrder, int * inputDimensions, int batchSize){

    input = new CRAbTensor[1];
    input[0] = CRAbTensor(inputOrder, inputDimensions, CT_NULL);

    output = new CRAbTensor[batchSize];
    for(int i = 0 ; i < batchSize ; i++)
        output[i] = CRAbTensor(inputOrder, inputDimensions, CT_ZEROS);

    backwardError = new CRAbTensor[batchSize];
    error = new CRAbTensor[batchSize];
    this->batchSize = batchSize;
    type = CNN_OUTPUT;


    //error = NULL;
    weightError = NULL;
    nextLayer = NULL;
    previousLayer = NULL;
}

void Output::sendError(){
    int inputOrder = input[0].getOrder();
    int * inputDimensions = input[0].getDimensions();

    // reshape error
    for(int i = 0 ; i < batchSize ; i++){
        error[i].reshape(inputOrder, inputDimensions);
        backwardError[i].deleteTensor();
        backwardError[i] = error[i] - output[i];
    }

    /*
    CRAbTensor temp, merror(backwardError[0].getOrder(), backwardError[0].getDimensions(), CT_ZEROS);
    for(int i = 0 ; i < batchSize ; i++){
        temp = merror + backwardError[i];
        merror.deleteTensor();
        merror = temp;
    }

    temp = merror * (1./((float)batchSize));
    merror.deleteTensor();
    merror = temp;
    */

    previousLayer->receiveError(backwardError);
    //merror.deleteTensor();
    previousLayer->sendError();
}

void Output::deleteLayer(){
    for(int i = 0 ; i < batchSize ; i++){
        input[i].deleteTensor();
        backwardError[i].deleteTensor();
    }

    delete [] input;
    delete [] backwardError;
}

void Output::receiveInput(CRAbTensor * input){
    for(int i = 0 ; i < batchSize ; i++)
        output[i].putdata(input[i].getdata());
}

void Output::receiveError(CRAbTensor * error){

    if(this->error != NULL)
        for(int i = 0 ; i < batchSize ; i++)
            this->error[i].deleteTensor();
    delete [] this->error;
    this->error = NULL;

    if(this->error == NULL)
        this->error = new CRAbTensor[batchSize];
    for(int i = 0 ; i < batchSize ; i++)
        this->error[i] = error[i].copy();
}
