//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#ifndef ALBANY_MESHADAPT_HPP
#define ALBANY_MESHADAPT_HPP

#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include "Albany_AbstractAdapter.hpp"
#include "Albany_FMDBMeshStruct.hpp"
#include "Albany_FMDBDiscretization.hpp"

#include "Phalanx.hpp"
#include "PHAL_Workset.hpp"
#include "PHAL_Dimension.hpp"

#include "Albany_UnifSizeField.hpp"
#include "Albany_UnifRefSizeField.hpp"


namespace Albany {

template<class SizeField>

class MeshAdapt : public AbstractAdapter {
public:

   MeshAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                     const Teuchos::RCP<ParamLib>& paramLib_,
                     Albany::StateManager& StateMgr_,
                     const Teuchos::RCP<const Epetra_Comm>& comm_);
   //! Destructor
    ~MeshAdapt();

    //! Check adaptation criteria to determine if the mesh needs adapting
    virtual bool queryAdaptationCriteria();

    //! Apply adaptation method to mesh and problem. Returns true if adaptation is performed successfully.
    // Solution is needed to calculate the size function
    virtual bool adaptMesh(const Epetra_Vector& solution, const Epetra_Vector& ovlp_solution);

    //! Transfer solution between meshes.
    virtual void solutionTransfer(const Epetra_Vector& oldSolution,
        Epetra_Vector& newSolution);

   //! Each adapter must generate it's list of valid parameters
    Teuchos::RCP<const Teuchos::ParameterList> getValidAdapterParameters() const;

private:

   // Disallow copy and assignment
   MeshAdapt(const MeshAdapt &);
   MeshAdapt &operator=(const MeshAdapt &);

   int numDim;
   int remeshFileIndex;

   Teuchos::RCP<Albany::FMDBMeshStruct> fmdbMeshStruct;

   Teuchos::RCP<Albany::AbstractDiscretization> disc;

   Albany::FMDBDiscretization *fmdb_discretization;

   pMeshMdl mesh;

   Teuchos::RCP<meshAdapt> rdr;
   int num_iterations;

   const Epetra_Vector* solution;
   const Epetra_Vector* ovlp_solution;

   static int setSizeField(pPart part, pSField field, void *vp);
   static Teuchos::RCP<SizeField> szField;

   void printElementData();

};

}

// Define macros for explicit template instantiation
#define MESHADAPT_INSTANTIATE_TEMPLATE_CLASS_UNIF(name) \
  template class name<Albany::UnifSizeField>;
#define MESHADAPT_INSTANTIATE_TEMPLATE_CLASS_UNIFREF(name) \
  template class name<Albany::UnifRefSizeField>;

#define MESHADAPT_INSTANTIATE_TEMPLATE_CLASS(name) \
  MESHADAPT_INSTANTIATE_TEMPLATE_CLASS_UNIF(name) \
  MESHADAPT_INSTANTIATE_TEMPLATE_CLASS_UNIFREF(name)


#endif //ALBANY_MESHADAPT_HPP
