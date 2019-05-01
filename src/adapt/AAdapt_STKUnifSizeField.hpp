//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef AADAPT_STK_UNIF_SIZE_FIELD_HPP
#define AADAPT_STK_UNIF_SIZE_FIELD_HPP

#include <stk_percept/PerceptMesh.hpp>
#include <stk_percept/function/ElementOp.hpp>

namespace AAdapt {

class STKUnifRefineField : public stk::percept::ElementOp {
public:

  STKUnifRefineField(stk::percept::PerceptMesh& eMesh) : m_eMesh(eMesh) {
  }

  virtual bool operator()(stk::mesh::Entity element,
                          stk::mesh::FieldBase* field,  const stk::mesh::BulkData& bulkData);
  virtual void init_elementOp() {}
  virtual void fini_elementOp() {}

private:
  stk::percept::PerceptMesh& m_eMesh;
};

class STKUnifUnrefineField : public stk::percept::ElementOp {

public:

  STKUnifUnrefineField(stk::percept::PerceptMesh& eMesh) : m_eMesh(eMesh) {
  }

  virtual bool operator()(stk::mesh::Entity element,
                          stk::mesh::FieldBase* field,  const stk::mesh::BulkData& bulkData);
  virtual void init_elementOp() {}
  virtual void fini_elementOp() {}

private:
  stk::percept::PerceptMesh& m_eMesh;
};

} // namespace AAdapt

#endif // AADAPT_STK_UNIF_SIZE_FIELD_HPP
