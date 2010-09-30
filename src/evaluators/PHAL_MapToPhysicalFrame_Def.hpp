/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"


//**********************************************************************
template<typename EvalT, typename Traits>
MapToPhysicalFrame<EvalT, Traits>::
MapToPhysicalFrame(const Teuchos::ParameterList& p) :
  coords_vertices  (p.get<std::string>                   ("Coordinate Vector Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("Coordinate Data Layout") ),
  coords_qp        (p.get<std::string>                   ("Coordinate Vector Name"),
	            p.get<Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout") ),
  cubature         (p.get<Teuchos::RCP <Intrepid::Cubature<RealType> > >("Cubature")),
  cellType         (p.get<Teuchos::RCP <shards::CellTopology> > ("Cell Type"))
{
  this->addDependentField(coords_vertices);
  this->addEvaluatedField(coords_qp);

  this->setName("MapToPhysicalFrame");
}

//**********************************************************************
template<typename EvalT, typename Traits>
void MapToPhysicalFrame<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coords_vertices,fm);
  this->utils.setFieldData(coords_qp,fm);

}

//**********************************************************************
template<typename EvalT, typename Traits>
void MapToPhysicalFrame<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // Grap cubature points using cubature object
  typename std::vector< typename PHX::template MDField<MeshScalarT,Cell,QuadPoint,Dim>::size_type > dims;
  coords_qp.dimensions(dims); //get dimensions

  // AGS: Consider pre-computing, or at least pre-allocating
  Intrepid::FieldContainer<RealType> refPoints(dims[1],dims[2]);
  Intrepid::FieldContainer<RealType> refWeights(dims[1]);
  cubature->getCubature(refPoints, refWeights); 
  
  Intrepid::CellTools<RealType>::mapToPhysicalFrame
       (coords_qp, refPoints, coords_vertices, *cellType);
}

//**********************************************************************
