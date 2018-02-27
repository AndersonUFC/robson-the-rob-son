#ifndef CNN_H
#define CNN_H

#include "input.h"
#include "batch.h"

#define CNN_STOCHASTIC_BATCH        0x0
#define CNN_FULL_BATCH              0x1
#define CNN_MINI_BATCH4             0x2
#define CNN_MINI_BATCH8             0x3
#define CNN_MINI_BATCH16            0x4
#define CNN_MINI_BATCH32            0x5
#define CNN_MINI_BATCH64            0x6
#define CNN_MINI_BATCH128           0x7

class CNN{
public:
    CNN(int tensorOrder , int * tensorDimensions, float learningRate, int batchType, int batchSize);

    // methods
    void predict();
    CRAbTensor predict(CRAbTensor input);
    void train();
    int * argmax();

    // layer creation
    void createReLU();
    void createConvolution(int * maskDimensions, int * padding, int numberOfChannels);
    void createPooling(int * maskDimensions, int * stride);
    void createFullConnected(int numberHiddenUnits);
    void createOutput();

    // seters & getters
    Layer * getInput();
    float getLearningRate();
    int getBatchMode();
    int getBatchSize();
    int getMiniBatchSize();
    int * getBatchList();
    Batch * getbatch();
    Layer * lastLayer();

    friend std::ostream  & operator << (std::ostream & os, CNN cnn){
        os << "CNN\n";
        Layer * iterator = cnn.input;

        while(iterator != NULL){
            os << Layer::printType(iterator->getType()) << " layer. Output shape ";
            iterator->getOutputTensor()[0].printShape();
            os << "\n";
            iterator = iterator->getNextLayer();
        }
        return os;
    }

private:

    // variables
    Layer * input;
    float learningRate;
    int batchMode, batchSize, miniBatchSize;
    Batch * batch;
    int * batchList;

    // methods
    int * generateBatchList();
};

namespace CNN_IO{
    // I/O
    void save(std::string name, CNN * cnn);
    CNN * load(std::string name);
}

#endif // CNN_H
