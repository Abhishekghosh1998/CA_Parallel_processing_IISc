#include <immintrin.h>
#define ALIGN 64 //cache size

void reference(int N, int* matA, int* matB, int* output);

void singleThread(int N, int* matA, int* matB, int* output){

 if(N<=4){
	reference(N, matA, matB, output);
	return;
 }

  int* b=(int*) aligned_alloc(ALIGN, sizeof(int)*N*N);

  //copying our matA and matB to the cache aligned memory locations
  int i,j;
  //copying matB to b in column major order.

  int k=0;
  for(j=0;j<N;j++)
    for(i=0;i<N;i++){
      b[k++]=matB[i*N+j];
    }

  //__m256i* A=(__m256i*) a;
  __m256i* B=(__m256i*) b;

  int n=N/8; //no of 256 bit words in a single row

  for(i=0;i<N;i+=2){
    for(j=0;j<N;j+=2){
      __m256i accumulator = _mm256_set1_epi32(0);
      __m256i temp,temp1;

      int chunk;
      for(chunk=0;chunk<n;chunk++){

        __m256i * ptr1 = (__m256i*)&matA[i*N+chunk*8];
        __m256i * ptr2 = (__m256i*)&matA[(i+1)*N+chunk*8];

        __m256i A_1= _mm256_loadu_si256(ptr1);
        __m256i A_2= _mm256_loadu_si256(ptr2);

	      temp=_mm256_add_epi32(A_1,A_2);
        temp1=_mm256_add_epi32(B[j*n+chunk],B[(j+1)*n+chunk]);
        temp1= _mm256_mullo_epi32(temp,temp1);
        accumulator=_mm256_add_epi32(temp1,accumulator);
      }

      accumulator = _mm256_hadd_epi32(accumulator,accumulator);
      accumulator = _mm256_hadd_epi32(accumulator,accumulator);
      __m256i acculumator_lane_reversed= _mm256_permute2f128_si256(accumulator, accumulator,1);

      accumulator = _mm256_add_epi32(acculumator_lane_reversed, accumulator);

      int rowC = i>>1;
      int colC = j>>1;
      int indexC = rowC * (N>>1) + colC;
      int * ptr=(int*)(&accumulator);
      output[indexC] = ptr[0];
    }
  }

}
