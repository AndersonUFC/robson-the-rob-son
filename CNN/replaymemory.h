#ifndef REPLAYMEMORY_H
#define REPLAYMEMORY_H

#include "crabtensor.h"


/**

  Experience: 5-tuple (state_1, action_1, reward, state_2, action_2)

  */
typedef struct Experience{
    CRAbTensor state1, state2;
    int action1, action2;
    CRAbTensor reward;

    bool empty;
}Experience;

class ReplayMemory{
public:
    ReplayMemory(int size);
    bool push(CRAbTensor state1, int action1, CRAbTensor reward, CRAbTensor state2, int action2);
    Experience pop();

    Experience * getAll();
    Experience operator [](int index);

    int memorySize();

    int * sample(int miniBatchSize);
    int * sampleByReward(int miniBatchSize);

private:
    Experience * list;
    int size, front, rear, queueSize;
};

#endif // REPLAYMEMORY_H
