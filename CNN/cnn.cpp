#include "cnn.h"

#include "relu.h"
#include "convolution.h"
#include "pooling.h"
#include "fullconnected.h"
#include "output.h"

CNN::CNN(int tensorOrder , int * tensorDimensions, float learningRate, int batchType, int batchSize){
    // set batch
    this->batchSize = batchSize;
    batchMode = batchType;
    switch(batchType){
        case CNN_STOCHASTIC_BATCH:
        miniBatchSize = 1;
        break;
        case CNN_FULL_BATCH:
        miniBatchSize = batchSize;
        break;
        case CNN_MINI_BATCH4:
        miniBatchSize = 4;
        break;
        case CNN_MINI_BATCH8:
        miniBatchSize = 8;
        break;
        case CNN_MINI_BATCH16:
        miniBatchSize = 16;
        break;
        case CNN_MINI_BATCH32:
        miniBatchSize = 32;
        break;
        case CNN_MINI_BATCH64:
        miniBatchSize = 64;
        break;
        case CNN_MINI_BATCH128:
        miniBatchSize = 128;
        break;
    }

    // create input layer
    Input * inp = new Input(tensorOrder, tensorDimensions, miniBatchSize);
    input = inp;

    // learning rate
    this->learningRate = learningRate;

    // batch
    batch = new Batch(batchSize);
    batchList = new int[1];
}

void CNN::predict(){
    delete [] batchList;
    batchList = generateBatchList();

    // generate minibatchdata
    CRAbTensor * miniBatch = new CRAbTensor[miniBatchSize];
    for(int i = 0 ; i < miniBatchSize ; i++){
	miniBatch[i].deleteTensor();
        miniBatch[i] = (*batch)[batchList[i]][0];
    }

    Layer * iterator = input;
    * iterator < miniBatch;

    delete [] miniBatch;

    while(iterator != NULL){
        for(int i = 0 ; i < miniBatchSize ; i++){

            iterator->run(i);
        }
        if(iterator->getNextLayer() != NULL)
            *iterator > iterator->getNextLayer();
        iterator = iterator->getNextLayer();
    }
}

CRAbTensor CNN::predict(CRAbTensor input_tensor){
    Layer * iterator = input;


    iterator->setBatchSize(1);
    * iterator < &input_tensor;
    iterator->setBatchSize(miniBatchSize);

    while(iterator != NULL){

        iterator->setBatchSize(1);
        iterator->run(0);
        if(iterator->getNextLayer() != NULL)
             *iterator > iterator->getNextLayer();
        iterator->setBatchSize(miniBatchSize);
        iterator = iterator->getNextLayer();

    }

    return lastLayer()->getOutputTensor()[0];
}

void CNN::train(){
    // generate minibatchdata error
    CRAbTensor * miniBatch = new CRAbTensor[miniBatchSize];
    for(int i = 0 ; i < miniBatchSize ; i++){
        miniBatch[i] = (*batch)[batchList[i]][1];
    }

    // send error to output to act as a loss layer
    Layer* last = lastLayer();
    last->receiveError(miniBatch);
    last->sendError();

    // update parameters
    input->update(learningRate);

    delete [] miniBatch;
}

int * CNN::argmax(){
    Layer* last = lastLayer();
    CRAbTensor * output = last->getOutputTensor();
    int tensorSize = output[0].getSize();

    int * arglist = new int[miniBatchSize];
    float * argvaluelist = new float[miniBatchSize];
    for(int i = 0 ; i < miniBatchSize ; i++){
        argvaluelist[i] = INT64_MIN;
        arglist[i] = 0;
    }

    for(int i = 0 ; i < miniBatchSize ; i++){
        for(int j = 0 ; j < tensorSize ; j++){
            if(output[i][j] > argvaluelist[i]){
                argvaluelist[i] = output[i][j];
                arglist[i] = j;
            }
        }
    }

    delete [] argvaluelist;
    return arglist;
}

// layer creation ---------------------------------------------------------------------------------
void CNN::createReLU(){
    Layer* layer = lastLayer();

    int order, *dimensions;
    CRAbTensor output = layer->getOutputTensor()[0];

    order = output.getOrder();
    dimensions = output.getDimensions();

    ReLU * relu = new ReLU(order, dimensions, miniBatchSize);
    Layer* newLayer = relu;

    *layer + newLayer;
}

void CNN::createConvolution(int * maskDimensions, int * padding, int numberOfChannels){
    Layer* layer = lastLayer();

    int order, *dimensions;
    CRAbTensor output = layer->getOutputTensor()[0];

    order = output.getOrder();
    dimensions = output.getDimensions();

    Convolution * conv = new Convolution(order, dimensions, maskDimensions, padding, numberOfChannels, miniBatchSize);
    Layer* newLayer = conv;

    *layer + newLayer;
}

void CNN::createPooling(int * maskDimensions, int * stride){
    Layer* layer = lastLayer();

    int order, *dimensions;
    CRAbTensor output = layer->getOutputTensor()[0];

    order = output.getOrder();
    dimensions = output.getDimensions();
    Pooling * pool = new Pooling(order, dimensions, maskDimensions, stride, miniBatchSize);
    Layer* newLayer = pool;

    *layer + newLayer;
}

void CNN::createFullConnected(int numberHiddenUnits){
    Layer* layer = lastLayer();

    int order, *dimensions;
    CRAbTensor output = layer->getOutputTensor()[0];

    order = output.getOrder();
    dimensions = output.getDimensions();
    FullConnected * fc = new FullConnected(order, dimensions, numberHiddenUnits, miniBatchSize);
    Layer* newLayer = fc;

    *layer + newLayer;
}

void CNN::createOutput(){
    Layer* layer = lastLayer();

    int order, *dimensions;
    CRAbTensor output = layer->getOutputTensor()[0];

    order = output.getOrder();
    dimensions = output.getDimensions();
    Output * out = new Output(order, dimensions, miniBatchSize);
    Layer* newLayer = out;

    *layer + newLayer;
}

// setters & getters ------------------------------------------------------------------------------

Layer * CNN::getInput(){
    return input;
}

float CNN::getLearningRate(){
    return learningRate;
}

int CNN::getBatchMode(){
    return batchMode;
}

int CNN::getBatchSize(){
    return batchSize;
}

int CNN::getMiniBatchSize(){
    return miniBatchSize;
}

Batch * CNN::getbatch(){
    return batch;
}

int * CNN::getBatchList(){
    return batchList;
}

// private methods --------------------------------------------------------------------------------

Layer * CNN::lastLayer(){
    Layer * last = input;
    while(last->getNextLayer() != NULL)
        last = last->getNextLayer();

    return last;
}

int * CNN::generateBatchList(){
    int bs = batch->getQueueSize();

    if(bs < miniBatchSize){
        std::cout << "batch error: too few batchs to process using current batch mode\n" <<
                     "current queue batch size: " << bs << "\n" <<
                     "current mini batch mode needed: " << miniBatchSize << "\n";
        exit(0);
    }

    int * list = new int[miniBatchSize];

    int * inside = new int[bs];
    for(int i = 0 ; i < bs ; i++)
        inside[i] = 0;

    int next;
    for(int i = 0 ; i < miniBatchSize ; i++){
        next = rand()%bs;

        while(inside[next] != 0)
            next = (next+1)%bs;

        inside[next] = 1;
        list[i] = next;
    }

    delete [] inside;
    return list;
}

// I/O --------------------------------------------------------------------------------------------

#include <iostream>
#include <fstream>
#include <ostream>

void CNN_IO::save(std::string name, CNN * cnn){
    std::ofstream outfile;
    outfile.open(name, std::ios::app);
    outfile.write((char*)cnn, sizeof(*cnn));
    outfile.close();
}

CNN * CNN_IO::load(std::string name){
    CNN * cnn;
    std::ifstream infile;
    infile.open(name, std::ios::in);
    infile.read((char*)cnn, sizeof(*cnn));

    return cnn;
}

