//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_STKUNIFSIZEFIELD_HPP
#define AADAPT_STKUNIFSIZEFIELD_HPP

#include <stk_percept/PerceptMesh.hpp>
#include <stk_percept/function/ElementOp.hpp>

namespace AAdapt {

class STKUnifRefineField : public stk_classic::percept::ElementOp {

  public:

    STKUnifRefineField(stk_classic::percept::PerceptMesh& eMesh) : m_eMesh(eMesh) {
    }

    virtual bool operator()(const stk_classic::mesh::Entity& element,
                            stk_classic::mesh::FieldBase* field,  const stk_classic::mesh::BulkData& bulkData);
    virtual void init_elementOp() {}
    virtual void fini_elementOp() {}

  private:
    stk_classic::percept::PerceptMesh& m_eMesh;
};

class STKUnifUnrefineField : public stk_classic::percept::ElementOp {

  public:

    STKUnifUnrefineField(stk_classic::percept::PerceptMesh& eMesh) : m_eMesh(eMesh) {
    }

    virtual bool operator()(const stk_classic::mesh::Entity& element,
                            stk_classic::mesh::FieldBase* field,  const stk_classic::mesh::BulkData& bulkData);
    virtual void init_elementOp() {}
    virtual void fini_elementOp() {}

  private:
    stk_classic::percept::PerceptMesh& m_eMesh;
};

}

#endif

