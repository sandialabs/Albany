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


#ifndef ENAT_SGNOXSOLVER_H
#define ENAT_SGNOXSOLVER_H

#include <iostream>

#include "LOCA.H"
#include "LOCA_Epetra.H"
#include "Epetra_Vector.h"
#include "Epetra_LocalMap.h"
#include "LOCA_Epetra_ModelEvaluatorInterface.H"
#include <NOX_Epetra_MultiVector.H>

#include "Albany_ModelEvaluator.hpp"
#include "Albany_Utils.hpp"
#include "Piro_Epetra_StokhosNOXObserver.hpp"

#include "Stokhos_CompletePolynomialBasis.hpp"
#include "Stokhos_Quadrature.hpp"

/** \brief Epetra-based Model Evaluator subclass for Charon!
 *
 * This class will support a wide number of different types of abstract
 * problem types that will allow NOX, LOCA, Rythmos, Aristos, and MOOCHO to
 * solve different types of problems with Charon.
 * 
 * ToDo: Finish documentation!
 */

namespace ENAT {
  class SGNOXSolver : public EpetraExt::ModelEvaluator {
  public:

    /** \name Constructors/initializers */
    //@{

    /** \brief Takes the number of elements in the discretization . */
    SGNOXSolver(const Teuchos::RCP<Teuchos::ParameterList>& appParams,
		const Teuchos::RCP<EpetraExt::ModelEvaluator>& model,
		const Teuchos::RCP<const Epetra_Comm>& comm,
                Teuchos::RCP<NOX::Epetra::Observer> noxObserver);


    //@}

    ~SGNOXSolver();


    /** \name Overridden from EpetraExt::ModelEvaluator . */
    //@{
    
    Teuchos::RCP<const Epetra_Map> get_g_map(int j) const;
    Teuchos::RCP<const Epetra_Map> get_g_sg_map(int j) const;
    /** \brief . */
    Teuchos::RCP<const Epetra_Vector> get_p_init(int l) const;
    Teuchos::RCP<const Stokhos::EpetraVectorOrthogPoly> get_p_sg_init(int l) const;
    /** \brief . */
    //  Teuchos::RCP<Epetra_Operator> create_W() const;
    /** \brief . */
    EpetraExt::ModelEvaluator::InArgs createInArgs() const;
    /** \brief . */
    EpetraExt::ModelEvaluator::OutArgs createOutArgs() const;
    /** \brief . */
    void evalModel( const InArgs& inArgs, const OutArgs& outArgs ) const;

    Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> >
    getBasis() const { return basis; }
    Teuchos::RCP<const Stokhos::Quadrature<int,double> >
    getQuad() const { return quad; }
    
  private:
    /** \brief . */
    Teuchos::RCP<const Epetra_Map> get_x_map() const;
    /** \brief . */
    Teuchos::RCP<const Epetra_Map> get_f_map() const;
    /** \brief . */
    Teuchos::RCP<const Epetra_Vector> get_x_init() const;
    /** \brief . */
    Teuchos::RCP<const Epetra_Map> get_p_map(int l) const;
    /** \brief . */
    Teuchos::RCP<const Epetra_Map> get_p_sg_map(int l) const;
    /** \brief . */
    void setProblemParamDefaults(Teuchos::ParameterList* appParams_);
    /** \brief . */
    void setSolverParamDefaults(Teuchos::ParameterList* appParams_);

    Teuchos::RCP<const Teuchos::ParameterList>
     getValidSGParameters() const;

    //@}
    
  private:
    
    enum SG_METHOD {
      SG_AD,
      SG_GLOBAL,
      SG_NI
    };

    //These are set in the constructor and used in evalModel
    Teuchos::RCP<EpetraExt::ModelEvaluator> sg_solver;
    Teuchos::RCP<const Stokhos::OrthogPolyBasis<int,double> > basis;
    Teuchos::RCP<const Stokhos::Quadrature<int,double> > quad;
  };
  
}
#endif
