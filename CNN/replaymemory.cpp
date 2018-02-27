#include "replaymemory.h"

ReplayMemory::ReplayMemory(int size){
    this->size = size;
    queueSize = 0;
    front = 0;
    rear = 0;

    list = new Experience[size];
}

bool ReplayMemory::push(CRAbTensor state1, int action1, CRAbTensor reward, CRAbTensor state2, int action2){
    if(queueSize < size){
        queueSize++;

        Experience exp;
        exp.state1 = state1.copy();
        exp.state2 = state2.copy();
        exp.action1 = action1;
        exp.action2 = action2;
        exp.reward = reward.copy();
        exp.empty = false;

        list[rear] = exp;

        rear = (rear+1)%size;
        return true;
    }
    return false;
}

Experience ReplayMemory::pop(){
    Experience exp;

    if(queueSize == 0){
        exp.empty = true;
    } else {
        queueSize--;
        exp = list[front];
        front= (front+1)%size;
    }

    return exp;
}

Experience * ReplayMemory::getAll(){
    return list;
}

Experience ReplayMemory::operator [](int index){
    if(index > size){
        std::cout << "replay memory error: index access of range (" << index << " >= " << size << ")\n";
        exit(0);
    }
    return list[(front + index)%size];
}

int ReplayMemory::memorySize(){
    return queueSize;
}

int * ReplayMemory::sample(int miniBatchSize){
    if(queueSize < miniBatchSize){
        std::cout << "replay memory error: too few batchs to process using current batch size\n" <<
                     "current queue batch size: " << queueSize << "\n" <<
                     "current mini batch size argument: " << miniBatchSize << "\n";
        exit(0);
    }

    int * list = new int[miniBatchSize];
    int * inside = new int[queueSize];

    for(int i = 0 ; i < queueSize ; i++)
        inside[i] = 0;

    int next;
    for(int i = 0 ; i < miniBatchSize ; i++){
        next = rand()%queueSize;

        while(inside[next] != 0)
            next = (next+1)%queueSize;

        inside[next] = 1;
        list[i] = next;
    }

    delete [] inside;
    return list;
}

int * ReplayMemory::sampleByReward(int miniBatchSize){
    if(queueSize < miniBatchSize){
        std::cout << "replay memory error: too few batchs to process using current batch size\n" <<
                     "current queue batch size: " << queueSize << "\n" <<
                     "current mini batch size argument: " << miniBatchSize << "\n";
        exit(0);
    }

    int * list = new int[miniBatchSize];
    int * inside = new int[queueSize];

    for(int i = 0 ; i < queueSize ; i++)
        inside[i] = 0;

    int n_rewards = 0;
    for(int i = 0 ; i < queueSize ; i++){
        if(this->list[i].reward != 0){
            inside[i] = 1;
            list[n_rewards] = i;
            n_rewards++;

            if(n_rewards >= miniBatchSize)
                break;
        }
    }

    int next;
    for(int i = n_rewards ; i < miniBatchSize ; i++){
        next = rand()%queueSize;

        while(inside[next] != 0)
            next = (next+1)%queueSize;

        inside[next] = 1;
        list[i] = next;
    }

    delete [] inside;
    return list;

}
