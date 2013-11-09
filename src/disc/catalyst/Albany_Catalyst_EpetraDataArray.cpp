//*****************************************************************//
//    Albany 2.0:  Copyright 2013 Kitware Inc.                     //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_Catalyst_EpetraDataArray.hpp"

#include <Shards_BasicTopologies.hpp>

#include <Teuchos_TestForException.hpp>

#include <vtkIdList.h>
#include <vtkIdTypeArray.h>
#include <vtkObjectFactory.h> // for vtkStandardNewMacro
#include <vtkVariantCast.h>

namespace Albany {
namespace Catalyst {

vtkStandardNewMacro(EpetraDataArray)

void EpetraDataArray::PrintSelf(ostream &os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

void EpetraDataArray::SetEpetraVector(const Epetra_Vector &vector)
{
  this->Data = &vector;
  this->NumberOfComponents = 1;
  this->Size = static_cast<vtkIdType>(vector.MyLength());
  this->MaxId = this->Size - 1;
}

void EpetraDataArray::Initialize()
{
  this->Data = NULL;
  this->Size = 0;
  this->MaxId = -1;
}

void EpetraDataArray::GetTuples(vtkIdList *ptIds, vtkAbstractArray *output)
{
  vtkDataArray *da = vtkDataArray::FastDownCast(output);
  TEUCHOS_TEST_FOR_EXCEPTION(!da, std::runtime_error,
                             "GetTuples must be given a vtkDataArray arg.");

  da->Reset();
  da->SetNumberOfComponents(1);
  da->Allocate(ptIds->GetNumberOfIds());

  vtkIdType *begin = ptIds->GetPointer(0);
  vtkIdType *end = ptIds->GetPointer(ptIds->GetNumberOfIds());
  while (begin != end)
    da->InsertNextTuple1((*this->Data)[*(begin++)]);
}

void EpetraDataArray::GetTuples(vtkIdType p1, vtkIdType p2,
                                vtkAbstractArray *output)
{
  vtkDataArray *da = vtkDataArray::FastDownCast(output);
  TEUCHOS_TEST_FOR_EXCEPTION(!da, std::runtime_error,
                             "GetTuples must be given a vtkDataArray arg.");

  da->Reset();
  da->SetNumberOfComponents(1);
  da->Allocate(p2 - p1 + 1);

  while (p1 <= p2)
    da->InsertNextTuple1((*this->Data)[p1++]);
}

void EpetraDataArray::Squeeze()
{
}

vtkArrayIterator *EpetraDataArray::NewIterator()
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
                             "Method not implemented.");
  return NULL;
}

vtkIdType EpetraDataArray::LookupValue(vtkVariant value)
{
  bool valid = false;
  double tmp = vtkVariantCast<double>(value, &valid);
  if (valid)
    return this->Lookup(tmp, 0);
  return -1;
}

void EpetraDataArray::LookupValue(vtkVariant value, vtkIdList *ids)
{
  bool valid = false;
  double tmp = vtkVariantCast<double>(value, &valid);
  ids->Reset();
  if (valid) {
    vtkIdType index = 0;
    while ((index = this->Lookup(tmp, index)) >= 0)
      ids->InsertNextId(index++);
  }
}

vtkVariant EpetraDataArray::GetVariantValue(vtkIdType idx)
{
  return vtkVariant((*this->Data)[idx]);
}

void EpetraDataArray::ClearLookup()
{
  // no-op, no fast lookup implemented.
}

double *EpetraDataArray::GetTuple(vtkIdType i)
{
  this->TmpDouble = (*this->Data)[i];
  return &this->TmpDouble;
}

void EpetraDataArray::GetTuple(vtkIdType i, double *tuple)
{
  *tuple = (*this->Data)[i];
}

vtkIdType EpetraDataArray::LookupTypedValue(ValueType value)
{
  return this->Lookup(value, 0);
}

void EpetraDataArray::LookupTypedValue(ValueType value, vtkIdList *ids)
{
  ids->Reset();
  vtkIdType index = 0;
  while ((index = this->Lookup(value, index)) >= 0)
    ids->InsertNextId(index++);
}

int EpetraDataArray::Allocate(vtkIdType sz, vtkIdType ext)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
  return 0;
}

int EpetraDataArray::Resize(vtkIdType numTuples)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
  return 0;
}

void EpetraDataArray::SetNumberOfTuples(vtkIdType number)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

void EpetraDataArray::SetTuple(vtkIdType i, vtkIdType j,
                               vtkAbstractArray *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

void EpetraDataArray::SetTuple(vtkIdType i, const float *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

void EpetraDataArray::SetTuple(vtkIdType i, const double *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

void EpetraDataArray::InsertTuple(vtkIdType i, vtkIdType j,
                                  vtkAbstractArray *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

void EpetraDataArray::InsertTuple(vtkIdType i, const float *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

void EpetraDataArray::InsertTuple(vtkIdType i, const double *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

void EpetraDataArray::InsertTuples(vtkIdList *dstIds, vtkIdList *srcIds,
                                   vtkAbstractArray *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

vtkIdType EpetraDataArray::InsertNextTuple(vtkIdType j,
                                           vtkAbstractArray *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
  return -1;
}

vtkIdType EpetraDataArray::InsertNextTuple(const float *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
  return -1;
}

vtkIdType EpetraDataArray::InsertNextTuple(const double *source)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
  return -1;
}

void EpetraDataArray::DeepCopy(vtkAbstractArray *aa)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

void EpetraDataArray::DeepCopy(vtkDataArray *da)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

void EpetraDataArray::InterpolateTuple(vtkIdType i, vtkIdList *ptIndices,
                                       vtkAbstractArray *source,
                                       double *weights)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

void EpetraDataArray::InterpolateTuple(vtkIdType i, vtkIdType id1,
                                       vtkAbstractArray *source1, vtkIdType id2,
                                       vtkAbstractArray *source2, double t)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

void EpetraDataArray::SetVariantValue(vtkIdType idx, vtkVariant value)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

void EpetraDataArray::RemoveTuple(vtkIdType id)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

void EpetraDataArray::RemoveFirstTuple()
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

void EpetraDataArray::RemoveLastTuple()
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

void EpetraDataArray::SetTupleValue(vtkIdType i, const ValueType *t)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

void EpetraDataArray::InsertTupleValue(vtkIdType i, const ValueType *t)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

vtkIdType EpetraDataArray::InsertNextTupleValue(const ValueType *t)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
  return -1;
}

void EpetraDataArray::SetValue(vtkIdType idx, ValueType value)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

vtkIdType EpetraDataArray::InsertNextValue(ValueType v)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
  return -1;
}

void EpetraDataArray::InsertValue(vtkIdType idx, ValueType v)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error, "Read-only container.");
}

EpetraDataArray::EpetraDataArray()
  : Data(NULL)
{
}

EpetraDataArray::~EpetraDataArray()
{
}

vtkIdType EpetraDataArray::Lookup(double val, vtkIdType index)
{
  while (index <= this->MaxId) {
    if ((*this->Data)[index] == val)
      return index;
    ++index;
  }
  return -1;
}

} // namespace Catalyst
} // namespace Albany
