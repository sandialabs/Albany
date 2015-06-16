//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef GOAL_DISCRETIZATION_HPP
#define GOAL_DISCRETIZATION_HPP

#include "Albany_DataTypes.hpp"

namespace apf {
class Mesh;
class Field;
}

namespace Albany {
class StateManager;
class PUMIMeshStruct;
class AbstractPUMIDiscretization;
}

namespace GOAL {

class Discretization
{
  public:
    Discretization(Teuchos::RCP<Albany::StateManager>& sm);
    ~Discretization();
    void writeMesh(const char* n);
    void updateSolutionToMesh(const Tpetra_Vector& x);
    void enrichDiscretization();
    void decreaseDiscretization();
    void fillSolution(Teuchos::RCP<Tpetra_Vector>& x);
    Teuchos::RCP<Albany::AbstractPUMIDiscretization>
      getPUMIDisc() { return disc; }
  private:
    apf::Mesh* mesh;
    std::vector<std::string> solNames;
    std::vector<int> solIndex;
    Teuchos::ArrayRCP<apf::Field*> solFields;
    Teuchos::RCP<Albany::AbstractPUMIDiscretization> disc;
};

}

#endif
