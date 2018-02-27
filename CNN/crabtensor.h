#ifndef CRABTENSOR_H
#define CRABTENSOR_H

#include "crabtensordef.h"
#include <iostream>

class CRAbTensor
{
public:
    // empty constructor
    CRAbTensor(){dimensions=NULL; accumulator=NULL; data=NULL;order=0;size=0;}
    CRAbTensor(int order, int * dimensions, int feedType);

    // setters & getters --------------------------------------------------------------------------
    float get(int * dimension);
    int getOrder();
    int* getDimensions();
    float* getdata();
    int* getAccumulator();
    int getSize();
    long getSizeBytes();
    int fromDimensionToInt(int * dimension);
    void reset(int order, int * dimensions, int feedType);

    void putdata(float* data);

    // operators ----------------------------------------------------------------------------------
    float & operator [] (int index);
    CRAbTensor operator + (CRAbTensor tensor);
    CRAbTensor operator - (CRAbTensor tensor);
    CRAbTensor operator * (CRAbTensor tensor);
    bool operator == (CRAbTensor tensor);
    bool operator == (int * dimensions);
    bool operator != (CRAbTensor tensor);
    bool operator != (int * dimensions);
    CRAbTensor operator + (float operand);
    CRAbTensor operator - (float operand);
    CRAbTensor operator * (float operand);
    void operator =(CRAbTensor t);

    // other methods ------------------------------------------------------------------------------
    bool consistency(CRAbTensor otherTensor);
    void reshape(int order, int * dimension);
    void printShape();
    void printData();
    void deleteTensor();
    void normalize();
    int argmax();
    float dot(CRAbTensor t);
    CRAbTensor copy();


    friend std::ostream& operator << (std::ostream& os, CRAbTensor tensor){
        if(tensor.data == NULL){
            os << "null tensor\n";
            return os;
        }
        os << "Tensor of order " << tensor.order << "\n";
        os << "Size: " << tensor.size << "\n";
        os << "Dimensions: ( ";
        for(int i = 0 ; i < tensor.order ; i++)
            os << tensor.dimensions[i] << " ";
        os << ")\n";
        os << "Accumulator: ( ";
        for(int i = 0 ; i < tensor.order ; i++)
            os << tensor.accumulator[i] << " ";
        os << ")\n";


        os << "Data:\n";

        // iterate over data ======================================================================
        int position = -1, base = 0;
        int last = tensor.dimensions[tensor.order - 1];

        int* current_position = new int[tensor.order];
        for(int i = 0 ; i < tensor.order ; i++)
            current_position[i] = 0;

        while(position + 1 < tensor.size){
            std::cout << "( ";
            for(int i = 0 ; i < last ; i++){
                current_position[tensor.order - 1] = i;
                position = base + i;
                std::cout << tensor.data[position] << " ";
            }// for
            std::cout << ")\n";

            base += last;

            for(int i = tensor.order - 2 ; i >= 0 ; i--){
                if(current_position[i] + 1 < tensor.dimensions[i]){
                    current_position[i]++;
                    break;
                } else {
                    if(i > 0){
                        current_position[i-1]++;
                        current_position[i] = 0;

                        std::cout << "\n";

                        if(current_position[i-1] < tensor.dimensions[i])
                            break;
                    }// if
                }// if
            }// for
        }// while

        // delete and return ======================================================================
        delete [] current_position;


        return os;
    }// operator <<

private:
    int order, size;
    int *dimensions, *accumulator;
    float *data;
};

#endif // CRABTENSOR_H
