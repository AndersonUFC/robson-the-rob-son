#include "reinforcementlearner.h"

ReinforcementLearner::ReinforcementLearner(float discount_rate, int number_actions){
    this->discount_rate = discount_rate;
    cnn = NULL;

    int outputdim[1] = {number_actions};
    Q1 = CRAbTensor(1, outputdim,CT_ZEROS);

    delete [] outputdim;
}


void ReinforcementLearner::run(){


}

/*
void ReinforcementLearner::run(){


}

*/
