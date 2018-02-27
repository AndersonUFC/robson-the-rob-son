#include "layer.h"

Layer::Layer(){ }

// methods ====================================================================================
void Layer::receiveError(CRAbTensor * error){
    const int uni_batch = 1;
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

void Layer::sendOutput(Layer* l){
    if(output[0] != l->getInputShape()){
        std::cout << "Layer error:  between " << Layer::printType(type) << " and " << Layer::printType(l->getType()) << "\n";
        std::cout << "              Layers with different shapes: ";
        output[0].printShape();
        l->getInputTensor()[0].printShape();
        std::cout << "\n";
        exit(0);
    }
    l->receiveInput(output);
}

// operators ==================================================================================
void Layer::operator + (Layer* layer){
    nextLayer = layer;
    layer->previousLayer = this;
}
void Layer::operator > (Layer* layer){
    sendOutput(layer);
}
void Layer::operator < (CRAbTensor * input){
    receiveInput(input);
}

// setters & getters ==========================================================================
CRAbTensor * Layer::getInputTensor(){
    return input;
}
CRAbTensor Layer::getWeightTensor(){
    return weights;
}
CRAbTensor * Layer::getOutputTensor(){
    return output;
}
int* Layer::getInputShape(){
    return input[0].getDimensions();
}
int* Layer::getWeightShape(){
    return weights.getDimensions();
}
int* Layer::getOutputShape(){
    return output[0].getDimensions();
}
Layer* Layer::getNextLayer(){
    return nextLayer;
}
Layer* Layer::getPreviousLayer(){
    return previousLayer;
}

int Layer::getBatchSize(){
    return batchSize;
}

int Layer::getType(){
    return type;
}

void Layer::setBatchSize(int batch){
    this->batchSize = batch;
}
