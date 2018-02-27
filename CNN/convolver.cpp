#include "convolver.h"

#include <cmath>

Convolver::Convolver(int numberOfChannels){
    numberChannels = numberOfChannels;

    int * bias_dim = new int[1];

    bias_dim[0] = numberOfChannels;

    bias.deleteTensor();
    bias = CRAbTensor(1, bias_dim, CT_RANDOM_POS);

    delete [] bias_dim;
}

void Convolver::convolve(int inputBatch){
    iterator.Delete();
    iterator = ConvolutionOperation(&(input[inputBatch]), &weights, &bias, true);
    iterator.iterate();

    output[inputBatch].deleteTensor();
    output[inputBatch] = iterator.getOutput();
}

void Convolver::run(int currentBatch){
    convolve(currentBatch);
}

void Convolver::update(float learningRate){
    const int uni_batch = 1;
    CRAbTensor meanError(weightError[0].getOrder(), weightError[0].getDimensions(), CT_ZEROS), temp, temp2;

    for(int i = 0 ; i < uni_batch ; i++){
        temp = meanError + weightError[i];
        meanError.deleteTensor();
        meanError = temp;
    }


    temp = meanError * (1./((float)uni_batch));
    meanError.deleteTensor();
    meanError = temp;

    temp = meanError * learningRate;
    meanError.deleteTensor();

    temp2 = weights + temp;

    temp.deleteTensor();
    weights.deleteTensor();

    weights = temp2;

    //updateBias(learningRate);

    if(nextLayer != NULL)
        nextLayer->update(learningRate);
}

void Convolver::deleteLayer(){
    for(int i = 0 ; i < batchSize ; i++){
        backwardError[i].deleteTensor();
        error[i].deleteTensor();
        input[i].deleteTensor();
        output[i].deleteTensor();
        weightError[i].deleteTensor();
    }

    delete [] backwardError;
    delete [] error;
    delete [] input;
    delete [] output;
    delete [] weightError;

    weights.deleteTensor();
}

void Convolver::updateBias(float learning_rate){

    const int uni_batch = 1;
    int * error_dimensions = error[0].getDimensions();
    int * error_accumulator = error[0].getAccumulator();
    int * mask_dimensions = weights.getDimensions();
    int filter_size = error_dimensions[0]/mask_dimensions[0];
    int error_size = error[0].getSize();
    int bias_size = bias.getSize();

    CRAbTensor * bias_ = new CRAbTensor[uni_batch];
    for(int i1 = 0 ; i1 < uni_batch ; i1++)
        bias_[i1] = CRAbTensor(1, bias.getDimensions(), CT_ZEROS);

    for(int i1 = 0 ; i1 < uni_batch ; i1++){
        for(int i2 = 0 ; i2 < error_size ; i2++){
            int current_filter = i2/error_accumulator[0];
            int currentu_channel = current_filter/filter_size;

            bias_[i1][currentu_channel] += error[i1][i2];
        }

        for(int i2 = 0 ; i2 < bias_size ; i2++)
            bias_[i1][i2] /= error_accumulator[0];
    }

    CRAbTensor bias_error = CRAbTensor(1, bias.getDimensions(), CT_ZEROS);

    for(int i1 = 0 ; i1 < uni_batch ; i1++)
        for(int i2 = 0 ; i2 < bias_size ; i2++)
            bias_error[i2] += bias_[i1][i2]/bias_size;

    for(int i = 0 ; i < bias_size ; i++){
        if( bias_error[i] > 0 && bias_error[i] < 1e-7){
            bias_error[i] = 0;
        } else if (bias_error[i] < 0 && bias_error[i]*(-1) < 1e-7){
            bias_error[i] = 0;
        }
    }

    CRAbTensor b2 = bias_error*learning_rate;
    CRAbTensor b3 = bias + b2;

    bias.deleteTensor();
    bias = b3.copy();

    // free pointers
    for(int i = 0 ; i < uni_batch ; i++)
        bias_[i].deleteTensor();
    delete [] bias_;

    bias_error.deleteTensor();
    b2.deleteTensor();
    b3.deleteTensor();
}
