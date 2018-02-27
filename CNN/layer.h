#ifndef LAYER_H
#define LAYER_H

#include "crabtensor.h"

#define CNN_INPUT                   0x0
#define CNN_RELU                    0x1
#define CNN_CONVOLUTION             0x2
#define CNN_POOLING                 0x3
#define CNN_FULL_CONNECTED          0x4
#define CNN_OUTPUT                  0x5
#define CNN_LOSS                    0x6
#define CNN_NONLINEARIZER           0x7

class Layer{
public:
    // empty constructor
    Layer();

    // methods ====================================================================================
    virtual void receiveError(CRAbTensor * error);
    void sendOutput(Layer* l);

    // virtual methods ============================================================================
    virtual void run(int currentBatch){}
    virtual void sendError(){}
    virtual void update(float learningRate){
        if(nextLayer != NULL)
            nextLayer->update(learningRate);
    }
    virtual void deleteLayer(){}
    virtual long getSizeBytes(){}
    virtual void receiveInput(CRAbTensor * input){}

    // operators ==================================================================================
    void operator + (Layer* layer);
    void operator > (Layer* layer);
    void operator < (CRAbTensor * input);

    // setters & getters ==========================================================================
    CRAbTensor * getInputTensor();
    CRAbTensor getWeightTensor();
    CRAbTensor * getOutputTensor();
    virtual int* getInputShape();
    int* getWeightShape();
    int* getOutputShape();
    Layer* getNextLayer();
    Layer* getPreviousLayer();
    int getBatchSize();
    int getType();
    void setBatchSize(int batch);

    static std::string printType(int typ){
        switch(typ){
            case CNN_INPUT:
            return "input";
            case CNN_RELU:
            return "relu";
            case CNN_CONVOLUTION:
            return "convolution";
            case CNN_POOLING:
            return "pooling";
            case CNN_FULL_CONNECTED:
            return "full connected";
            case CNN_OUTPUT:
            return "output";
            case CNN_LOSS:
            return "loss";
        }
        return "";
    }

protected:
    CRAbTensor *input, weights, *output, *error, *weightError, *backwardError;
    int batchSize, type;
    Layer *nextLayer, *previousLayer;
};

#endif // LAYER_H
