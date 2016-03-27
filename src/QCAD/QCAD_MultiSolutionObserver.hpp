//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef QCAD_MULTISOLUTIONOBSERVER_HPP
#define QCAD_MULTISOLUTIONOBSERVER_HPP

#include "Epetra_Vector.h"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Albany_Application.hpp"

namespace QCAD {

  // Utility functions
  void CopyAllStates(Albany::StateArrays& src, Albany::StateArrays& dest,
		     const Teuchos::RCP<const Albany::StateInfoStruct>& stateInfo);

  void separateCombinedVector(Teuchos::RCP<const Epetra_Map> disc_map,
			      int numCopiesOfDiscMap, int numAdditionalElements,
			      const Teuchos::RCP<const Epetra_Comm>& comm,
			      const Teuchos::RCP<Epetra_Vector>& combinedVector,
			      Teuchos::RCP<Epetra_MultiVector>& disc_parts,
			      Teuchos::RCP<Epetra_Vector>& additional_part);

  Teuchos::RCP<Epetra_Map> CreateCombinedMap(Teuchos::RCP<const Epetra_Map> disc_map,
					     int numCopiesOfDiscMap, int numAdditionalElements,
					     const Teuchos::RCP<const Epetra_Comm>& comm);




  //! Multi-solution observer: allows output to exodus in which the fields from multiple 
  //   Albany::Application objects are combined, and also supports outputting eigenvectors as nodal
  //    fields in the exodus file.  It is assumed that all apps have the same discretization map,
  //    and so state and field data is compatible.
class MultiSolution_Observer
{
public:
  MultiSolution_Observer (const Teuchos::RCP<Albany::Application>& app,
			  const Teuchos::RCP<Teuchos::ParameterList>& params); //single app constructor

  MultiSolution_Observer (const Teuchos::RCP<Albany::Application>& app1, 
			  const Teuchos::RCP<Albany::Application>& app2,
			  const Teuchos::RCP<Teuchos::ParameterList>& params); //two-app constructor
  // TODO: make more constructors: (three-app, arbitrary app, etc)

  ~MultiSolution_Observer () { };

  void observeSolution( const Epetra_Vector& solution, const std::string& solutionLabel,
			Teuchos::RCP<Albany::EigendataStruct> eigenData = Teuchos::null,
			double stamp = 0.0);
  // TODO: make more observeSolutions: (two-solution, arbitrary solution, etc)

private:
  Teuchos::RCP<Teuchos::ParameterList> rootParams;
  std::vector< Teuchos::RCP<Albany::Application> > apps;
};



} // namespace QCAD

#endif

