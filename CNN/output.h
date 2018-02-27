#ifndef OUTPUT_H
#define OUTPUT_H

#include "layer.h"

class Output : public Layer
{
public:
    Output(int inputOrder, int * inputDimensions, int batchSize);
    void sendError();
    void deleteLayer();
    void receiveError(CRAbTensor* error);
    void receiveInput(CRAbTensor * input);
};

#endif // OUTPUT_H

/*

    virtual void sendError(){}
    virtual void deleteLayer(){}
    virtual void receiveInput(CRAbTensor * input){}

 */
