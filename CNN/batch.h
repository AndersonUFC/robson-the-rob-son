#ifndef BATCH_H
#define BATCH_H

#include "crabtensor.h"
#include <iostream>

class Batch{
public:
    Batch(int size);

    bool push(CRAbTensor tensorInput, CRAbTensor tensorOutput);
    CRAbTensor* pop();
    CRAbTensor** get();

    friend std::ostream & operator << (std::ostream & os, Batch b){
        os << "[" << b.front << "  " << b.rear << "]\n";
        return os;
    }
    CRAbTensor * operator [](int index);
    int getQueueSize();

    void clear();
private:
    int size, front, rear, queueSize;
    CRAbTensor** batch;
};

#endif // BATCH_H
