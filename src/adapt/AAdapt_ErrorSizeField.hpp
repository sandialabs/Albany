//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

/*******************************************************************/
/*           THIS WILL ONLY WORK FOR LINEAR TETS                   */
/*******************************************************************/

#ifndef AADAPT_ERRORSIZEFIELD_HPP
#define AADAPT_ERRORSIZEFIELD_HPP

#include "Albany_FMDBDiscretization.hpp"
#include "Epetra_Vector.h"
#include "AdaptTypes.h"
#include "MeshAdapt.h"

namespace MATH {
typedef std::vector<double> vec;
typedef std::vector<vec> matrix;
}

namespace AAdapt {

class ErrorSizeField {

  public:
    ErrorSizeField(Albany::FMDBDiscretization* disc);
    ~ErrorSizeField();

    void setParams(const Epetra_Vector* sol, const Epetra_Vector* ovlp_sol, double element_size);
    void setError();

    int computeSizeField(pPart part, pSField field);

  private:

    void computeVertexMeshSize(pPart part);
    void computeElementalMeshSize(pPart part);

    void initializeErrorTag(pPart part);
    void computeError(pPart part);

    double computeTetSize(pMeshEnt tet);
    double computeTetVolume(pMeshEnt tet);
    MATH::vec computeTetStrain(pMeshEnt tet, pTag disp_tag);

    double computeL2Norm(MATH::vec x);
    MATH::vec multiplyMatrixVec(MATH::matrix A, MATH::vec b);
    MATH::matrix computeInverse3x3(MATH::matrix A);

    Albany::FMDBDiscretization* disc;
    Teuchos::RCP<Albany::FMDBMeshStruct> mesh_struct;
    pMeshMdl mesh;

    const Epetra_Vector* solution;
    const Epetra_Vector* ovlp_solution;
    double elem_size;

    pTag disp_tag;

    pTag error_tag;
    double global_min_err;
    double global_max_err;

    pTag elem_h_new;
    pTag vtx_h_new;

};

}

#endif

