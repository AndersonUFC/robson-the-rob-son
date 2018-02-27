#include "batch.h"

// constructor ------------------------------------------------------------------------------------

Batch::Batch(int size){
    this->size = size;
    queueSize = 0;
    front = 0;
    rear =  0;
    batch = new CRAbTensor*[size];
    for(int i = 0 ; i < size ; i++)
        batch[i] = new CRAbTensor[2];
}

// methods ----------------------------------------------------------------------------------------

bool Batch::push(CRAbTensor input, CRAbTensor output){
    if(queueSize < size){

        queueSize++;
        batch[rear][0] = input.copy();
        batch[rear][1] = output.copy();
        rear = (rear+1)%size;
        return true;
    }// if
    return false;
}

CRAbTensor* Batch::pop(){
    CRAbTensor* a = NULL;
    if(queueSize == 0)
	return NULL;
    queueSize--;

    a = batch[front];
    front = (front+1)%size;
    return a;
}

CRAbTensor** Batch::get(){
    return batch;
}

int Batch::getQueueSize(){
    return queueSize;
}

void Batch::clear(){
    while(queueSize > 0){
        CRAbTensor * k = pop();
        if(k != NULL){
            k[0].deleteTensor();
            k[1].deleteTensor();
        }
    }
}

// operators --------------------------------------------------------------------------------------

CRAbTensor * Batch::operator [](int index){
    if(index > size){
        std::cout << "batch error: index access of range (" << index << " >= " << size << ")\n";
        exit(0);
    }
    return batch[(front + index)%size];
}

