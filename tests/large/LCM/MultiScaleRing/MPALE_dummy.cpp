/* 
  MPALE example 
  Glen Hansen, gahanse@sandia.gov

  This code receives a stress value from Albany and just sends it back.

*/ 
 
#include "mpi.h" 
#include <cstdlib>
#include <iostream>
#include <vector>


enum {STRESS_TENSOR, STRAIN_TENSOR, TANGENT, DIE};

void print_error(std::string stuff){

    std::cerr << stuff << std::endl;
    exit(-1);

}

void print_vec(std::vector<double>& vec){

  for(int i = 0; i < 3; i++){
    std::cout << " [ ";
    for(int j = 0; j < 3; j++)

      std::cout << vec[i*3 + j] << " ";
    std::cout << "] " << std::endl;
  }
  std::cout << std::endl;
}


int main(int argc, char *argv[]) { 

   int size; 
   MPI_Comm albany; 
   MPI_Init(&argc, &argv); 
   MPI_Comm_get_parent(&albany); 
   MPI_Status status;

   int my_mpale_number;
   MPI_Comm_rank(albany, &my_mpale_number);

   if (albany == MPI_COMM_NULL) 
    print_error("No parent!"); 

   MPI_Comm_remote_size(albany, &size); 

   std::vector<double> albany_stress_tensor(9);

   std::vector<double> mpale_stress_tensor(9);


   if (size != 1) 
    print_error("Something's wrong with the parent"); 

    while(true){

        // Get the stress tensor
    
        MPI_Recv(&albany_stress_tensor[0], albany_stress_tensor.size(), MPI_DOUBLE, 0, MPI_ANY_TAG,
                 albany, &status);

        // TIme to die?

        if (status.MPI_TAG == DIE) {
           return 0;
        }

    
        // Do the calculations
    
        mpale_stress_tensor = albany_stress_tensor;
    
        // Send it back out
    
        MPI_Send(&mpale_stress_tensor[0], mpale_stress_tensor.size(), MPI_DOUBLE, 0, STRESS_TENSOR, albany);

    }

    MPI_Finalize(); 
    return 0; 
} 
