#include <pthread.h>
#define MAX_THREADS 64
// Create other necessary functions here

void reference(int N, int* matA, int* matB, int* output);

struct argument{
    int N;
    int* matA;
    int* matB;
    int* matC;
    int i; 
};

void* dot_product(void * args){

    struct argument* ptr = (struct argument*) args;
    int* A= ptr->matA;
    int* B= ptr->matB;
    int* C= ptr->matC;
    int index=ptr->i;
    int N= ptr->N;

    int length= (N>>1)*(N>>1);
    int n=N/8;
    int i;
    for(i=index;i<length;i+=MAX_THREADS){
        int row_index=i/(N>>1);
        int col_index= i%(N>>1);

        int rowA=row_index<<1;
        int colB=col_index<<1;

      __m256i accumulator = _mm256_set1_epi32(0);
      __m256i temp,temp1;

      int chunk;
      for(chunk=0;chunk<n;chunk++){

        __m256i * ptr1 = (__m256i*)&A[rowA*N+chunk*8]; 
        __m256i * ptr2 = (__m256i*)&A[(rowA+1)*N+chunk*8];

        __m256i A_1= _mm256_loadu_si256(ptr1);
        __m256i A_2= _mm256_loadu_si256(ptr2);

	    __m256i * ptr3 = (__m256i*)&B[colB*N+chunk*8]; 
        __m256i * ptr4 = (__m256i*)&B[(colB+1)*N+chunk*8];

        __m256i B_1= _mm256_loadu_si256(ptr3);
        __m256i B_2= _mm256_loadu_si256(ptr4);

	    temp=_mm256_add_epi32(A_1,A_2);
        temp1=_mm256_add_epi32(B_1,B_2);
        temp1= _mm256_mullo_epi32(temp,temp1);
        accumulator=_mm256_add_epi32(temp1,accumulator); 

      }

      accumulator = _mm256_hadd_epi32(accumulator,accumulator);
      accumulator = _mm256_hadd_epi32(accumulator,accumulator);
      __m256i acculumator_lane_reversed= _mm256_permute2f128_si256(accumulator, accumulator,1);

      accumulator = _mm256_add_epi32(acculumator_lane_reversed, accumulator);
      C[i]=*(int*)(&accumulator);


    }
    free(ptr);

    return NULL;
}

// Fill in this function
void multiThread(int N, int *matA, int *matB, int *output)
{
    if(N<=4){
	reference(N, matA, matB, output);
	return ;
	}
    int *A=matA;
    int *B=(int*)malloc(sizeof(int)*N*N);
    /* The following copy can also be done in parallel to get boost up*/
    int i,j,k=0; 
    for(j=0;j<N;j++)
        for(i=0;i<N;i++)
            B[k++]=matB[i*N+j];
    
    pthread_t t[MAX_THREADS];
    
    int length=(N>>1)*(N>>1);
    for(i=0;i<MAX_THREADS && i < length ;i++){

        struct argument* ptr= (struct argument*) malloc(sizeof(struct argument));
        ptr->matA=A;
        ptr->matB=B;
        ptr->matC=output;
        ptr->i=i;
	ptr->N=N;

        if(pthread_create(t+i,NULL, dot_product, (void*)ptr)){
            std::cout<<"Failed create thread "<<i<<std::endl;
            exit(1);
        }
    }

    for(i=0;i<MAX_THREADS && i<length;i++){
        if(pthread_join(t[i],NULL)){
            std::cout<<"Failed to thread "<<i<<std::endl;
            exit(1);
        }
    }
}

