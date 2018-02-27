#include <stdlib.h>

#include "crabtensor.h"

CRAbTensor::CRAbTensor(int order, int * dimensions, int feedType){
    this->order = order;
    this->dimensions = new int[order];
    this->accumulator = new int[order];

    int total_size = 1;
    for(int i = 0 ; i < order ; i++)
        this->dimensions[i] = dimensions[i];

    for(int i = 0 ; i < order ; i++){
        total_size *= this->dimensions[i];
    }// for
    this->size = total_size;
    data = new float[total_size];

    if(feedType == CT_ZEROS){
        for(int i = 0 ; i < total_size ; i++){
            data[i] = 0;
        }// for
    } else if (feedType == CT_RANDOM){
        for(int i = 0 ; i < total_size ; i++){
            data[i] = (float(rand())*((rand()%2)*2 - 1))/(float(RAND_MAX));
        }// for
    } else if (feedType == CT_RANDOM_POS){
        for(int i = 0 ; i < total_size ; i++){
            data[i] = float(rand())/(float(RAND_MAX));
        }// for
    } else if (feedType == CT_SMALL_INTEGER){
        for(int i = 0 ; i < total_size ; i++)
            data[i] = rand()%5;
    } else if (feedType == CT_BIN){
        for(int i = 0 ; i < total_size ; i++)
            data[i] = rand()%2;
    } else if (feedType == CT_MIN_NEG){
        for(int i = 0 ; i < total_size ; i++)
            data[i] = INT64_MIN;
    }// if-kaskade

    accumulator[order-1] = 1;
    for(int i = order-2 ; i >=0 ; i--){
        accumulator[i] = this->dimensions[i+1]*accumulator[i+1];
    }// for
}// constructor

// SETTERS & GETTERS

void CRAbTensor::reset(int order, int * dimensions, int feedType){
    this->order = order;

    if(this->dimensions != NULL)
        delete [] dimensions;
    this->dimensions = new int[order];

    int total_size = 1;
    for(int i = 0 ; i < order ; i++)
        this->dimensions[i] = dimensions[i];

    for(int i = 0 ; i < order ; i++){
        total_size *= this->dimensions[i];
    }// for

    this->size = total_size;
    if(data != NULL)
        delete [] data;
    this->data = new float[total_size];

    if(feedType == CT_ZEROS){
        for(int i = 0 ; i < total_size ; i++){
            data[i] = 0;
        }// for
    } else if (feedType == CT_RANDOM){
        for(int i = 0 ; i < total_size ; i++){
            data[i] = (float(rand())*((rand()%2)*2 - 1))/(float(RAND_MAX));
        }// for
    } else if (feedType == CT_RANDOM_POS){
        for(int i = 0 ; i < total_size ; i++){
            data[i] = float(rand())/(float(RAND_MAX));
        }// for
    } else if (feedType == CT_SMALL_INTEGER){
        for(int i = 0 ; i < total_size ; i++)
            data[i] = rand()%5;
    } else if (feedType == CT_BIN){
        for(int i = 0 ; i < total_size ; i++)
            data[i] = rand()%2;
    } else if (feedType == CT_MIN_NEG){
        for(int i = 0 ; i < total_size ; i++)
            data[i] = INT64_MIN;
    }// if-kaskade

    if(accumulator != NULL)
        delete [] accumulator;
    accumulator = new int[order];

    accumulator[order-1] = 1;
    for(int i = order-2 ; i >=0 ; i--){
        accumulator[i] = this->dimensions[i+1]*accumulator[i+1];
    }// for
}

float CRAbTensor::get(int * dimension){
    int index = fromDimensionToInt(dimension);
    return data[index];
}

int CRAbTensor::getOrder(){
    return order;

}

int* CRAbTensor::getDimensions(){
    return dimensions;
}

float* CRAbTensor::getdata(){
    return data;
}

int* CRAbTensor::getAccumulator(){
    return accumulator;
}

int CRAbTensor::getSize(){
    return size;
}

long CRAbTensor::getSizeBytes(){
    return size*(sizeof(float)) + order*2*(sizeof(int)) + 2*(sizeof(int));
}

void CRAbTensor::putdata(float* data){
    for(int i = 0 ; i < size ; i++)
        this->data[i] = data[i];
}

int CRAbTensor::fromDimensionToInt(int * dimension){
    int index = 0;


    for(int i = 0 ; i < order ; i++)
        index += dimension[i]*accumulator[i];

    return index;
}

// Operators --------------------------------------------------------------------------------------

float & CRAbTensor::operator [] (int index){
    if(index >= size){
        std::cout << "Data access index out of range ( " << index << " >= " << size << ")\n";
        exit(0);
    }// if

    return data[index];
}// operator []

CRAbTensor CRAbTensor::operator + (CRAbTensor t){
    if(!consistency(t)){
        std::cout << "Tensors with different dimensions: ";
        printShape();
        t.printShape();
        std::cout << "\n";
        exit(0);
    }// if

    CRAbTensor new_t = CRAbTensor(order, t.getDimensions(), CT_NULL);

    float* new_data = new float[size];
    float* t_data = t.getdata();

    for(int i = 0 ; i < size ; i++)
        new_data[i] = data[i] + t_data[i];

    new_t.putdata(new_data);

    delete [] new_data;
    new_data = NULL;

    return new_t;
}// operator +

CRAbTensor CRAbTensor::operator - (CRAbTensor t){
    if(!consistency(t)){
        std::cout << "Tensors with different dimensions: ";
        printShape();
        t.printShape();
        std::cout << "\n";
        exit(0);
    }// if

    CRAbTensor new_t = CRAbTensor(order, t.getDimensions(), CT_NULL);

    float* new_data = new float[size];
    float* t_data = t.getdata();

    for(int i = 0 ; i < size ; i++)
        new_data[i] = data[i] - t_data[i];

    new_t.putdata(new_data);

    delete [] new_data;
    new_data = NULL;

    return new_t;
}// operator -

CRAbTensor CRAbTensor::operator * (CRAbTensor t){
    if(!consistency(t)){
        std::cout << "Tensors with different dimensions: ";
        printShape();
        t.printShape();
        std::cout << "\n";
        exit(0);
    }// if

    CRAbTensor new_t = CRAbTensor(order, t.getDimensions(), CT_NULL);

    float* new_data = new float[size];
    float* t_data = t.getdata();

    for(int i = 0 ; i < size ; i++)
        new_data[i] = data[i] * t_data[i];

    new_t.putdata(new_data);

    delete [] new_data;
    new_data = NULL;

    return new_t;
}

CRAbTensor CRAbTensor::operator + (float n){
    CRAbTensor new_t = CRAbTensor(order, dimensions, CT_NULL);
    float* new_data = new float[size];

    for(int i = 0 ; i < size ; i++)
        new_data[i] = data[i] + n;

    new_t.putdata(new_data);

    delete [] new_data;
    new_data = NULL;

    return new_t;
}

CRAbTensor CRAbTensor::operator - (float n){
    CRAbTensor new_t = CRAbTensor(order, dimensions, CT_NULL);
    float* new_data = new float[size];

    for(int i = 0 ; i < size ; i++)
        new_data[i] = data[i] - n;

    new_t.putdata(new_data);

    delete [] new_data;
    new_data = NULL;

    return new_t;
}

CRAbTensor CRAbTensor::operator * (float n){
    CRAbTensor new_t = CRAbTensor(order, dimensions, CT_NULL);
    float* new_data = new float[size];

    for(int i = 0 ; i < size ; i++)
        new_data[i] = data[i] * n;

    new_t.putdata(new_data);

    delete [] new_data;
    new_data = NULL;

    return new_t;
}

bool CRAbTensor::operator ==(CRAbTensor tensor){
    return consistency(tensor);
}// shape equality

bool CRAbTensor::operator == (int * dim){
    for(int i = 0 ; i < order ; i++){
        if(dim[i] != dimensions[i])
            return false;
    }// for
    return true;
}

bool CRAbTensor::operator !=(CRAbTensor tensor){
    return !consistency(tensor);
}// shape inequality


bool CRAbTensor::operator != (int * dim){
    return !(*this == dim);
}


void CRAbTensor::operator =(CRAbTensor t){
    dimensions = t.getDimensions();
    accumulator = t.getAccumulator();
    order = t.getOrder();
    size = t.getSize();
    data= t.getdata();
}

// Other Methods ----------------------------------------------------------------------------------
bool CRAbTensor::consistency(CRAbTensor otherTensor){
    if(otherTensor.getOrder() != order)
        return false;

    int* dim = otherTensor.getDimensions();
    for(int i = 0 ; i < order ; i++){
        if(dim[i] != dimensions[i])
            return false;
    }// for

    return true;
}// consistency

void CRAbTensor::reshape(int dim_size, int * new_dim){
    int total = 1;
    for(int i = 0 ; i < dim_size ; i++)
        total *= new_dim[i];

    if(total != size){
        std::cout << "Reshape error: new shape with different total size (" << total << " != " << size << ")\n";
        exit(0);
    }// if

    delete [] dimensions;
    dimensions = new int[dim_size];

    for(int i = 0 ; i < dim_size ; i++)
        dimensions[i] = new_dim[i];

    delete [] accumulator;
    accumulator = new int[dim_size];

    this->order = dim_size;

    accumulator[order-1] = 1;
    for(int i = order-2 ; i >=0 ; i--){
        accumulator[i] = dimensions[i+1]*accumulator[i+1];
    }// for
}

void CRAbTensor::printShape(){
    std::cout << "( ";
    for(int i = 0 ; i < order ; i++)
        std::cout << dimensions[i] << " ";
    std::cout << ")";
}

void CRAbTensor::deleteTensor(){
    if(this->data != NULL){
        delete [] this->data;
        this->data = NULL;
    }
    if(this->dimensions != NULL){
        delete [] this->dimensions;
        this->dimensions = NULL;
    }
    if(this->accumulator!= NULL){
        delete [] this->accumulator;
        this->accumulator = NULL;
    }
}

CRAbTensor CRAbTensor::copy(){
    CRAbTensor a(order, dimensions, CT_NULL);
    for(int i = 0 ; i < size ; i++)
        a[i] = data[i];
    return a;
}

void CRAbTensor::printData(){
    std::cout << "[ ";
    for(int i = 0 ; i < size ; i++)
        std::cout << data[i] << " ";
    std::cout << "]\n";
}

void CRAbTensor::normalize(){

}

int CRAbTensor::argmax(){
    int index = 0;
    float value = INT64_MIN;
    for(int i = 0 ; i < size ; i++){
        if(data[i] > value){
            value = data[i];
            index = i;
        }
    }

    return index;
}


float CRAbTensor::dot(CRAbTensor t){
    float result = 0;

    for(int i = 0 ; i < size ; i++)
        result += data[i] * t[i];

    return result;
}
