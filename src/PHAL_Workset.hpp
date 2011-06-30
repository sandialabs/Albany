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


#ifndef PHAL_WORKSET_HPP
#define PHAL_WORKSET_HPP

#include <list>

#include "Phalanx_ConfigDefs.hpp" // for std::vector
#include "Albany_DataTypes.hpp" 
#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"
#include "Albany_AbstractDiscretization.hpp"
#include "Albany_StateManager.hpp"
#include "Stokhos_OrthogPolyExpansion.hpp"
#include "Stokhos_VectorOrthogPoly.hpp"
#include "Stokhos_VectorOrthogPolyTraitsEpetra.hpp"
#include <Intrepid_FieldContainer.hpp>
#include "PHAL_AlbanyTraits.hpp"

namespace PHAL {

struct Workset {
  
  Workset() :
    transientTerms(false), ignore_residual(false) {}

  unsigned int numCells;
  unsigned int worksetSize;

  Teuchos::RCP<Stokhos::OrthogPolyExpansion<int,double> > sg_expansion;

  Teuchos::RCP<const Epetra_Vector> x;
  Teuchos::RCP<const Epetra_Vector> xdot;
  Teuchos::RCP<ParamVec> params;
  Teuchos::RCP<const Epetra_MultiVector> Vx;
  Teuchos::RCP<const Epetra_MultiVector> Vxdot;
  Teuchos::RCP<const Epetra_MultiVector> Vp;
  Teuchos::RCP<const Stokhos::VectorOrthogPoly<Epetra_Vector> > sg_x;
  Teuchos::RCP<const Stokhos::VectorOrthogPoly<Epetra_Vector> > sg_xdot;
  Teuchos::RCP<const Stokhos::ProductContainer<Epetra_Vector> > mp_x;
  Teuchos::RCP<const Stokhos::ProductContainer<Epetra_Vector> > mp_xdot;

  Teuchos::RCP<Epetra_Vector> f;
  Teuchos::RCP<Epetra_CrsMatrix> Jac;
  Teuchos::RCP<Epetra_MultiVector> JV;
  Teuchos::RCP<Epetra_MultiVector> fp;
  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_Vector> > sg_f;
  Teuchos::RCP< Stokhos::VectorOrthogPoly<Epetra_CrsMatrix> > sg_Jac;
  Teuchos::RCP< Stokhos::ProductContainer<Epetra_Vector> > mp_f;
  Teuchos::RCP< Stokhos::ProductContainer<Epetra_CrsMatrix> > mp_Jac;

  Teuchos::RCP<const Albany::NodeSetList> nodeSets;
  Teuchos::RCP<const Albany::NodeSetCoordList> nodeSetCoords;

  // jacobian and mass matrix coefficients for matrix fill
  double j_coeff;
  double m_coeff;

  // Current Time as defined by Rythmos
  double current_time;

  // flag indicating whether to sum tangent derivatives, i.e.,
  // compute alpha*df/dxdot*Vxdot + beta*df/dx*Vx + df/dp*Vp or
  // compute alpha*df/dxdot*Vxdot + beta*df/dx*Vx and df/dp*Vp separately
  int num_cols_x;
  int num_cols_p;
  int param_offset;

  std::vector<int> *coord_deriv_indices;

  Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> >  wsElNodeID;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> >  wsCoords;
  Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > > >  ws_coord_derivs;
  std::string EBName;

  Teuchos::RCP<const Albany::StateVariables> oldState;
  Teuchos::RCP<Albany::StateVariables> newState;
  Albany::StateArray* stateArrayPtr;
  Albany::EigendataStruct* eigenDataPtr;

  bool transientTerms;

  // Flag indicating whether to ignore residual calculations in the 
  // Jacobian calculation.  This only works for some problems where the 
  // the calculation of the Jacobian doesn't require calculation of the
  // residual (such as linear problems), but if it does work it can
  // significantly reduce Jacobian calculation cost.
  bool ignore_residual;

  // Flag indicated whether we are solving the adjoint operator or the
  // forward operator.  This is used in the Albany application when
  // either the Jacobian or the transpose of the Jacobian is scattered. 
  bool is_adjoint;

};

}

#endif
