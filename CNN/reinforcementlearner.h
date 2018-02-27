#ifndef REINFORCEMENTLEARNER_H
#define REINFORCEMENTLEARNER_H

#include "cnn.h"

class ReinforcementLearner{
public:
    ReinforcementLearner(float discount_rate, int number_actions);

    // operations
    void run();

    // setters & getters
    CNN * getCNN(){return cnn;}


private:
    CNN * cnn;
    CRAbTensor Q1, Q2;
    float discount_rate, replay_memory_size;
};

#endif // REINFORCEMENTLEARNER_H
