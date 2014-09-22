//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ATO_OPTIMIZATION_PROBLEM_HPP
#define ATO_OPTIMIZATION_PROBLEM_HPP

#include "Albany_AbstractProblem.hpp"
#include "Epetra_Vector.h"

namespace ATO {

class Topology;

class OptimizationProblem :
public virtual Albany::AbstractProblem {

  public:
   OptimizationProblem( const Teuchos::RCP<Teuchos::ParameterList>& _params,
                        const Teuchos::RCP<ParamLib>& _paramLib,
                        const int _numDim);


   void ComputeVolume(const double* p, double& v, double* dvdp=NULL);
   void ComputeVolume(double& v);
   void setDiscretization(Teuchos::RCP<Albany::AbstractDiscretization> _disc)
          {disc = _disc;}
   void setCommunicator(const Teuchos::RCP<const Epetra_Comm>& _comm) {comm = _comm;}


   void InitTopOpt();

  protected:
   void setupTopOpt( Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> >  _meshSpecs,
                     Albany::StateManager& _stateMgr);

   Teuchos::RCP<Albany::AbstractDiscretization> disc;
   Teuchos::RCP<const Epetra_Comm> comm;

   Teuchos::ArrayRCP<Teuchos::RCP<Albany::MeshSpecsStruct> > meshSpecs;
   Albany::StateManager* stateMgr;

   std::vector<Teuchos::RCP<shards::CellTopology> > cellTypes;
   std::vector<Teuchos::RCP<Intrepid::Cubature<double> > > cubatures;
   std::vector<Teuchos::RCP<Intrepid::Basis<RealType, Intrepid::FieldContainer<RealType> > > >
     intrepidBasis;


   std::vector<Intrepid::FieldContainer<double> > refPoints;
   std::vector<Intrepid::FieldContainer<double> > refWeights;
   std::vector<Intrepid::FieldContainer<double> > basisAtQPs;
   std::vector<Intrepid::FieldContainer<double> > weighted_measure;

   Teuchos::RCP<Epetra_Vector> overlapVec;
   Teuchos::RCP<Epetra_Vector> localVec;
   Teuchos::RCP<Epetra_Export> exporter;

   Teuchos::RCP<Topology> topology;
};

}

#endif
