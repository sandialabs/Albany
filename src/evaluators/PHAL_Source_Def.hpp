//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <cmath>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "Sacado_ParameterAccessor.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Albany_Utils.hpp"
#ifdef ALBANY_STOKHOS
#include "Stokhos_KL_ExponentialRandomField.hpp"
#endif
#include "Teuchos_Array.hpp"
#include "Teuchos_TestForException.hpp"

namespace PHAL {

namespace Source_Functions {
const double pi = 3.1415926535897932385;

template <typename EvalT, typename Traits>
class Source_Base {
public :
  Source_Base(){};
  virtual ~Source_Base(){};
  virtual void EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p)                     = 0;
  virtual void DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p)                     = 0;
  virtual void FieldData      (PHX::EvaluatorUtilities<EvalT,Traits> &utils, PHX::FieldManager<Traits>& fm) = 0;
  virtual void evaluateFields (typename Traits::EvalData workset)                                         = 0;
};

///////////////////////////////////////////////////////////////////////////////

template<typename EvalT, typename Traits>
class Constant : 
    public Source_Base<EvalT,Traits>, 
    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
public :
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  static bool check_for_existance(Teuchos::ParameterList* source_list);
  Constant(Teuchos::ParameterList& p);
  virtual ~Constant(){}
  virtual void EvaluatedFields(Source<EvalT,Traits> &source, 
			       Teuchos::ParameterList& p);
  virtual void DependentFields(Source<EvalT,Traits> &source, 
			       Teuchos::ParameterList& p);
  virtual void FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, 
			 PHX::FieldManager<Traits>& fm);
  virtual void evaluateFields (typename Traits::EvalData workset);
  virtual ScalarT & getValue(const std::string &n) { return m_constant;};
private :
  ScalarT     m_constant;
  std::size_t m_num_qp;
  Teuchos::ParameterList* m_source_list;
  PHX::MDField<ScalarT,Cell,Point> m_source;
};

template<typename EvalT,typename Traits>
bool 
Constant<EvalT,Traits>::
check_for_existance(Teuchos::ParameterList* source_list)
{
  const bool exists = source_list->getEntryPtr("Constant");
  return exists;
}

template<typename EvalT,typename Traits>
Constant<EvalT,Traits>::
Constant(Teuchos::ParameterList& p) {
  m_source_list = p.get<Teuchos::ParameterList*>("Parameter List", NULL);
  Teuchos::ParameterList& paramList = m_source_list->sublist("Constant");
  m_constant = paramList.get("Value", 0.0);
  // Add the factor as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib> > ("Parameter Library", Teuchos::null);
  this->registerSacadoParameter("Constant Source Value", paramLib);
}

template<typename EvalT,typename Traits>
void Constant<EvalT,Traits>::
EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  Teuchos::RCP<PHX::DataLayout> dl = 
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Source Name"), dl);
  m_source = f ;
  source.addEvaluatedField(m_source);
}

template<typename EvalT,typename Traits>
void 
Constant<EvalT,Traits>::
DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
}

template<typename EvalT,typename Traits>
void 
Constant<EvalT,Traits>::
FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, 
	  PHX::FieldManager<Traits>& fm){
  utils.setFieldData(m_source, fm);
  typename std::vector< typename PHX::template MDField<ScalarT,Cell,Node>::size_type > dims;
  m_source.dimensions(dims);
  m_num_qp = dims[1];
}

template<typename EvalT,typename Traits>
void 
Constant<EvalT,Traits>::
evaluateFields(typename Traits::EvalData workset){

  // Loop over cells, quad points: compute Constant Source Term
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t iqp=0; iqp<m_num_qp; iqp++)
      m_source(cell, iqp) = m_constant;
  }
}


///////////////////////////////////////////////////////////////////////////////

template<typename EvalT, typename Traits>
class Table : 
    public Source_Base<EvalT,Traits>, 
    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
public :
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  static bool check_for_existance(Teuchos::ParameterList* source_list);
  Table(Teuchos::ParameterList& p);
  virtual ~Table(){}
  virtual void EvaluatedFields(Source<EvalT,Traits> &source, 
			       Teuchos::ParameterList& p);
  virtual void DependentFields(Source<EvalT,Traits> &source, 
			       Teuchos::ParameterList& p);
  virtual void FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, 
			 PHX::FieldManager<Traits>& fm);
  virtual void evaluateFields (typename Traits::EvalData workset);
  virtual ScalarT & getValue(const std::string &n) { return m_constant;};
private :
  ScalarT     m_constant;
  std::size_t m_num_qp;
  Teuchos::ParameterList* m_source_list;
  PHX::MDField<ScalarT,Cell,Point> m_source;
  std::vector<double> time;
  std::vector<double> sourceval;
  int num_time_vals;
};

template<typename EvalT,typename Traits>
bool 
Table<EvalT,Traits>::
check_for_existance(Teuchos::ParameterList* source_list)
{
  const bool exists = source_list->getEntryPtr("Table");
  return exists;
}

template<typename EvalT,typename Traits>
Table<EvalT,Traits>::
Table(Teuchos::ParameterList& p) {
  m_source_list = p.get<Teuchos::ParameterList*>("Parameter List", NULL);
  Teuchos::ParameterList& paramList = m_source_list->sublist("Table");
  std::string filename = paramList.get("Filename", "missing");

  // open file
  std::ifstream inFile(&filename[0]); 

  TEUCHOS_TEST_FOR_EXCEPTION(!inFile, Teuchos::Exceptions::InvalidParameter, std::endl <<
		     "Error! Cannot open tabular data file \"" << filename 
		     << "\" in source table fill" << std::endl);

  // Count lines in file
  int array_size = std::count(std::istreambuf_iterator<char>(inFile), 
             std::istreambuf_iterator<char>(), '\n');

  // Allocate and fill arrays
  time.resize(array_size);
  sourceval.resize(array_size);

  // rewind file
  inFile.seekg(0);

  for(num_time_vals = 0; num_time_vals < array_size; num_time_vals++){

    if(inFile.eof()) break;
    inFile >> time[num_time_vals];
    if(inFile.eof()) break;
    inFile >> sourceval[num_time_vals];

//std::cout << "time " << num_time_vals << " is " << time[num_time_vals] 
// << " " << sourceval[num_time_vals] << std::endl;
  }

  inFile.close();

  m_constant = sourceval[0];

  // Add the factor as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib> > ("Parameter Library", Teuchos::null);
  this->registerSacadoParameter("Table Source Value", paramLib);
}

template<typename EvalT,typename Traits>
void Table<EvalT,Traits>::
EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  Teuchos::RCP<PHX::DataLayout> dl = 
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Source Name"), dl);
  m_source = f ;
  source.addEvaluatedField(m_source);
}

template<typename EvalT,typename Traits>
void 
Table<EvalT,Traits>::
DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
}

template<typename EvalT,typename Traits>
void 
Table<EvalT,Traits>::
FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, 
	  PHX::FieldManager<Traits>& fm){
  utils.setFieldData(m_source, fm);
  typename std::vector< typename PHX::template MDField<ScalarT,Cell,Node>::size_type > dims;
  m_source.dimensions(dims);
  m_num_qp = dims[1];
}

template<typename EvalT,typename Traits>
void 
Table<EvalT,Traits>::
evaluateFields(typename Traits::EvalData workset){

  if(workset.current_time <= 0.0) // if time is uninitialized or zero, just take first value

    m_constant = sourceval[0];

  else { // Interpolate between time values

    bool found_it = false;

    for(int i = 0; i < num_time_vals - 1; i++) // Stride through time

      if(workset.current_time >= time[i] && workset.current_time <= time[i + 1] ){ // Have bracketed current time

        double s = (workset.current_time - time[i]) / (time[i + 1] - time[i]); // 0 \leq s \leq 1

        m_constant = sourceval[i] + s * (sourceval[i + 1] - sourceval[i]); // interp value corresponding to s

        found_it = true;

        break;

      }

    TEUCHOS_TEST_FOR_EXCEPTION(!found_it, Teuchos::Exceptions::InvalidParameter, std::endl <<
		     "Error! Cannot locate the current time \"" << workset.current_time 
		     << "\" in the time series data between the endpoints " << time[0]
          << " and " << time[num_time_vals - 1] << "." << std::endl);
  }

  // Loop over cells, quad points: compute Table Source Term
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t iqp=0; iqp<m_num_qp; iqp++)
      m_source(cell, iqp) = m_constant;
  }
}

////////////////////////////////////////////////////////////////////////////////


template<typename EvalT, typename Traits>
class Trigonometric : 
    public Source_Base<EvalT,Traits>, 
    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
public :
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  static bool check_for_existance(Teuchos::ParameterList* source_list);
  Trigonometric(Teuchos::ParameterList& p);
  virtual ~Trigonometric(){}
  virtual void EvaluatedFields(Source<EvalT,Traits> &source, 
			       Teuchos::ParameterList& p);
  virtual void DependentFields(Source<EvalT,Traits> &source, 
			       Teuchos::ParameterList& p);
  virtual void FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, 
			 PHX::FieldManager<Traits>& fm);
  virtual void evaluateFields (typename Traits::EvalData workset);
  virtual ScalarT & getValue(const std::string &n) { return m_constant;};
private :
  ScalarT     m_constant; 
  std::size_t m_num_qp;
  std::size_t m_num_dim;
  Teuchos::ParameterList* m_source_list;
  PHX::MDField<ScalarT,Cell,Point> m_source;
  PHX::MDField<MeshScalarT,Cell,Point,Dim> coordVec;
};

template<typename EvalT,typename Traits>
bool 
Trigonometric<EvalT,Traits>::
check_for_existance(Teuchos::ParameterList* source_list)
{
  const bool exists = source_list->getEntryPtr("Trigonometric");
  return exists;
}

template<typename EvalT,typename Traits>
Trigonometric<EvalT,Traits>::
Trigonometric(Teuchos::ParameterList& p) {
  m_source_list = p.get<Teuchos::ParameterList*>("Parameter List", NULL);
  Teuchos::ParameterList& paramList = m_source_list->sublist("Trigonometric");
  m_constant = paramList.get("Value", 1.0); 
  // Add the factor as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib> > ("Parameter Library", Teuchos::null);
  this->registerSacadoParameter("Trigonometric Source Value", paramLib);
}

template<typename EvalT,typename Traits>
void Trigonometric<EvalT,Traits>::
EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  Teuchos::RCP<PHX::DataLayout> dl = 
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Source Name"), dl);
  m_source = f ;
  source.addEvaluatedField(m_source);
}

template<typename EvalT,typename Traits>
void 
Trigonometric<EvalT,Traits>::
DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) 
{
  Teuchos::RCP<PHX::DataLayout> scalar_qp = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  Teuchos::RCP<PHX::DataLayout> vector_qp = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");  

  PHX::MDField<MeshScalarT,Cell,Point,Dim> f0
    (p.get<std::string>("QP Coordinate Vector Name"),  vector_qp);
  coordVec = f0;
  source.addDependentField(coordVec.fieldTag());
}

template<typename EvalT,typename Traits>
void 
Trigonometric<EvalT,Traits>::
FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, 
	  PHX::FieldManager<Traits>& fm){
  utils.setFieldData(m_source, fm);
  utils.setFieldData(coordVec,fm);
  typename std::vector< typename PHX::template MDField<ScalarT,Cell,Node>::size_type > dims;
  m_source.dimensions(dims);
  coordVec.dimensions(dims); 
  m_num_qp = dims[1];
  m_num_dim = dims[2]; 
}

template<typename EvalT,typename Traits>
void 
Trigonometric<EvalT,Traits>::
evaluateFields(typename Traits::EvalData workset){

  // Loop over cells, quad points: compute Trigonometric Source Term
  if (m_num_dim == 2) {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t iqp=0; iqp<m_num_qp; iqp++) {
        //MeshScalarT *X = &coordVec(cell,iqp,0); 
        m_source(cell, iqp) = 8.0*pi*pi*sin(2.0*pi*coordVec(cell,iqp,0))*cos(2.0*pi*coordVec(cell,iqp,1));
      }
    }
  }
  else {
    std::cout << "Trigonometric source implemented only for 2D; setting f = 1 constant source." << std::endl; 
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t iqp=0; iqp<m_num_qp; iqp++) {
        m_source(cell, iqp) = m_constant;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////


template<typename EvalT, typename Traits>
class Quadratic : 
    public Source_Base<EvalT,Traits>, 
    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
public :
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  static bool check_for_existance(Teuchos::ParameterList* source_list);
  Quadratic(Teuchos::ParameterList& p);
  virtual ~Quadratic(){}
  virtual void EvaluatedFields(Source<EvalT,Traits> &source, 
			       Teuchos::ParameterList& p);
  virtual void DependentFields(Source<EvalT,Traits> &source, 
			       Teuchos::ParameterList& p);
  virtual void FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, 
			 PHX::FieldManager<Traits>& fm);
  virtual void evaluateFields (typename Traits::EvalData workset);
  virtual ScalarT & getValue(const std::string &n) { return m_factor;};
private :
  ScalarT     m_factor;
  double      m_constant;
  std::size_t m_num_qp;
  Teuchos::ParameterList* m_source_list;
  PHX::MDField<ScalarT,Cell,Point>   m_source;
  PHX::MDField<ScalarT,Cell,Point>   m_baseField;
};

template<typename EvalT,typename Traits>
bool 
Quadratic<EvalT,Traits>::
check_for_existance(Teuchos::ParameterList* source_list)
{
  const bool exists = source_list->getEntryPtr("Quadratic");
  return exists;
}

template<typename EvalT,typename Traits>
Quadratic<EvalT,Traits>::
Quadratic(Teuchos::ParameterList& p) {
  m_source_list = p.get<Teuchos::ParameterList*>("Parameter List", NULL);
  Teuchos::ParameterList& paramList = m_source_list->sublist("Quadratic");
  m_factor   = paramList.get("Nonlinear Factor", 0.0);
  m_constant = paramList.get("Constant", false);
  // Add the factor as a Sacado-ized parameter
  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib> > ("Parameter Library", Teuchos::null);
  this->registerSacadoParameter("Quadratic Nonlinear Factor", paramLib);
}

template<typename EvalT,typename Traits>
void 
Quadratic<EvalT,Traits>::
EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  Teuchos::RCP<PHX::DataLayout> dl = 
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Source Name"), dl);
  m_source = f ;
  source.addEvaluatedField(m_source);
}
template<typename EvalT,typename Traits>
void 
Quadratic<EvalT,Traits>::
DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  Teuchos::RCP<PHX::DataLayout> dl = 
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Variable Name"), dl);
  m_baseField = f;
  source.addDependentField(m_baseField.fieldTag());
}
template<typename EvalT,typename Traits>
void 
Quadratic<EvalT,Traits>::
FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, 
	  PHX::FieldManager<Traits>& fm){
  utils.setFieldData(m_source,   fm);
  utils.setFieldData(m_baseField,fm);
  typename std::vector< typename PHX::template MDField<ScalarT,Cell,Node>::size_type > dims;
  m_source.dimensions(dims);
  m_num_qp = dims[1];
}

template<typename EvalT,typename Traits>
void 
Quadratic<EvalT,Traits>::
evaluateFields(typename Traits::EvalData workset){

  // Loop over cells, quad points: compute Quadratic Source Term
  if (m_constant) {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t iqp=0; iqp<m_num_qp; iqp++)
        m_source(cell, iqp) = m_factor;
    }
  } 
  else {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
      for (std::size_t iqp=0; iqp<m_num_qp; iqp++)
        m_source(cell, iqp) = 
	  m_factor * m_baseField(cell,iqp) * m_baseField(cell,iqp);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

#ifdef ALBANY_STOKHOS
template<typename EvalT, typename Traits>
class TruncatedKL : 
    public Source_Base<EvalT,Traits>, 
    public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
public :
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  static bool check_for_existance(Teuchos::ParameterList* source_list);
  TruncatedKL(Teuchos::ParameterList& p);
  virtual ~TruncatedKL(){}
  virtual void EvaluatedFields(Source<EvalT,Traits> &source, 
			       Teuchos::ParameterList& p);
  virtual void DependentFields(Source<EvalT,Traits> &source, 
			       Teuchos::ParameterList& p);
  virtual void FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, 
			 PHX::FieldManager<Traits>& fm);
  virtual void evaluateFields (typename Traits::EvalData workset);
  virtual ScalarT & getValue(const std::string &n);
private :
  std::size_t m_num_qp;
  std::size_t m_num_dims;
  Teuchos::ParameterList* m_source_list;
  PHX::MDField<ScalarT,Cell,Point>   m_source;
  PHX::MDField<MeshScalarT,Cell,Point,Dim> m_coordVec;
  Teuchos::RCP< Stokhos::KL::ExponentialRandomField<MeshScalarT> > m_exp_rf_kl;
  Teuchos::Array<ScalarT> m_rv;
  Teuchos::Array<MeshScalarT> m_point;
  std::string param_name_base;
};

template<typename EvalT,typename Traits>
bool 
TruncatedKL<EvalT,Traits>::
check_for_existance(Teuchos::ParameterList* source_list)
{
  const bool exists = source_list->getEntryPtr("Truncated KL Expansion");
  return exists;
}

template<typename EvalT,typename Traits>
TruncatedKL<EvalT,Traits>::
TruncatedKL(Teuchos::ParameterList& p) {
  m_source_list = p.get<Teuchos::ParameterList*>("Parameter List", NULL);
  Teuchos::ParameterList& paramList = 
    m_source_list->sublist("Truncated KL Expansion");
  
  m_exp_rf_kl = 
      Teuchos::rcp(new Stokhos::KL::ExponentialRandomField<MeshScalarT>(paramList));
  int num_KL = m_exp_rf_kl->stochasticDimension();

  param_name_base = p.get<std::string>("Source Name") + " KL Random Variable";

  // Add KL random variables as Sacado-ized parameters
  m_rv.resize(num_KL);
  Teuchos::RCP<ParamLib> paramLib = 
    p.get< Teuchos::RCP<ParamLib> >("Parameter Library", Teuchos::null);
  for (int i=0; i<num_KL; i++) {
    std::string ss = Albany::strint(param_name_base,i);
    this->registerSacadoParameter(ss, paramLib);
    m_rv[i] = paramList.get(ss, 0.0);
  }
}

template<typename EvalT,typename Traits>
void 
TruncatedKL<EvalT,Traits>::
EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  Teuchos::RCP<PHX::DataLayout> dl = 
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Source Name"), dl);
  m_source = f ;
  source.addEvaluatedField(m_source);
}

template<typename EvalT,typename Traits>
void 
TruncatedKL<EvalT,Traits>::
DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  Teuchos::RCP<PHX::DataLayout> vector_dl =
    p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");
  PHX::MDField<MeshScalarT,Cell,Point,Dim>
    fx(p.get<std::string>("QP Coordinate Vector Name"), vector_dl);
  m_coordVec = fx;
  source.addDependentField(m_coordVec.fieldTag());
}

template<typename EvalT,typename Traits>
void 
TruncatedKL<EvalT,Traits>::
FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, 
	  PHX::FieldManager<Traits>& fm){
  utils.setFieldData(m_source,   fm);
  utils.setFieldData(m_coordVec,fm);
  typename std::vector< typename PHX::template MDField<ScalarT,Cell,Node>::size_type > dims;
  m_coordVec.dimensions(dims);
  m_num_qp = dims[1];
  m_num_dims = dims[2];
  m_point.resize(m_num_dims);
}

template<typename EvalT,typename Traits>
void 
TruncatedKL<EvalT,Traits>::
evaluateFields(typename Traits::EvalData workset){

  // Loop over cells, quad points: compute TruncatedKL Source Term
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t qp=0; qp<m_num_qp; qp++) {
      for (std::size_t i=0; i<m_num_dims; i++)
	m_point[i] = 
	  Sacado::ScalarValue<MeshScalarT>::eval(m_coordVec(cell,qp,i));
        m_source(cell, qp) = m_exp_rf_kl->evaluate(m_point, m_rv);
    }
  }
}

template<typename EvalT,typename Traits>
typename TruncatedKL<EvalT,Traits>::ScalarT& 
TruncatedKL<EvalT,Traits>::
getValue(const std::string &n) {
  for (int i=0; i<m_rv.size(); i++) {
    if (n == Albany::strint(param_name_base,i))
      return m_rv[i];
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidParameter,
		     std::endl <<
		     "Error! Logic error in getting parameter " << n
		     << " in TruncatedKL::getValue()" << std::endl);
}
#endif

////////////////////////////////////////////////////////////////////////////////

template<typename EvalT, typename Traits>
class NeutronFission : 
    public Source_Base<EvalT,Traits> {
public :
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  static bool check_for_existance(Teuchos::ParameterList* source_list) {
    return source_list->isSublist("Neutron Fission");
  }

  NeutronFission(Teuchos::ParameterList& p) {
    m_source_list = p.get<Teuchos::ParameterList*>("Parameter List", NULL);
  }

  virtual ~NeutronFission() {}

  virtual void EvaluatedFields(Source<EvalT,Traits> &source, 
			       Teuchos::ParameterList& p) {
    Teuchos::RCP<PHX::DataLayout> dl = 
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
    PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Source Name"), dl);
    m_source = f ;
    source.addEvaluatedField(m_source);
  }

  virtual void DependentFields(Source<EvalT,Traits> &source, 
			       Teuchos::ParameterList& p) {
    Teuchos::RCP<PHX::DataLayout> dl = 
      p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
    m_phi = PHX::MDField<ScalarT,Cell,Point>(
      p.get<std::string>("Neutron Flux Name"), dl);
    m_sigma_f = PHX::MDField<ScalarT,Cell,Point>(
      p.get<std::string>("Fission Cross Section Name"), dl);
    m_E_f = PHX::MDField<ScalarT,Cell,Point>(
      p.get<std::string>("Energy Released per Fission Name"), dl);
    source.addDependentField(m_phi.fieldTag());
    source.addDependentField(m_sigma_f.fieldTag());
    source.addDependentField(m_E_f.fieldTag());
  }

  virtual void FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, 
			 PHX::FieldManager<Traits>& fm) {
    utils.setFieldData(m_source,   fm);
    utils.setFieldData(m_phi,fm);
    utils.setFieldData(m_sigma_f,fm);
    utils.setFieldData(m_E_f,fm);
    typename std::vector< typename PHX::template MDField<ScalarT,Cell,Node>::size_type > dims;
    m_source.dimensions(dims);
    m_num_qp = dims[1];
  }

  virtual void evaluateFields (typename Traits::EvalData workset) {
    for (std::size_t cell = 0; cell < workset.numCells; ++cell)
      for (std::size_t iqp=0; iqp<m_num_qp; iqp++)
        m_source(cell, iqp) = 
	  m_phi(cell,iqp) * m_sigma_f(cell,iqp) * m_E_f(cell,iqp);
  }

private :
  std::size_t m_num_qp;
  Teuchos::ParameterList* m_source_list;
  PHX::MDField<ScalarT,Cell,Point>   m_source;
  PHX::MDField<ScalarT,Cell,Point>   m_phi;
  PHX::MDField<ScalarT,Cell,Point>   m_sigma_f;
  PHX::MDField<ScalarT,Cell,Point>   m_E_f;
};

////////////////////////////////////////////////////////////////////////////////

template<typename EvalT, typename Traits>
class MVQuadratic : public Source_Base<EvalT,Traits>, public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
public :
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  static bool check_for_existance(Teuchos::ParameterList* source_list);
  MVQuadratic(Teuchos::ParameterList& p);
  virtual ~MVQuadratic(){}
  virtual void EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p);
  virtual void DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p);
  virtual void FieldData      (PHX::EvaluatorUtilities<EvalT,Traits> &utils, PHX::FieldManager<Traits>& fm);
  virtual void evaluateFields (typename Traits::EvalData workset);
  virtual ScalarT & getValue(const std::string &n);
private :
  std::vector<ScalarT>     m_factor;
  std::size_t m_num_qp;
  Teuchos::ParameterList* m_source_list;
  PHX::MDField<ScalarT,Cell,Point>   m_source;
  PHX::MDField<ScalarT,Cell,Point>   m_baseField;
};

template<typename EvalT,typename Traits>
bool MVQuadratic<EvalT,Traits>::check_for_existance(Teuchos::ParameterList* source_list)
{
  const bool exists = source_list->getEntryPtr("Multivariate Quadratic");
  return exists;
}

template<typename EvalT,typename Traits>
MVQuadratic<EvalT,Traits>::MVQuadratic(Teuchos::ParameterList& p) {
  m_source_list = p.get<Teuchos::ParameterList*>("Parameter List", NULL);
  Teuchos::ParameterList& paramList = m_source_list->sublist("Multivariate Quadratic");
  int num_vars = paramList.get("Dimension", 1);
  m_factor.resize(num_vars);
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> > ("Parameter Library", Teuchos::null);
  for (int i=0; i<num_vars; i++) {
    m_factor[i] = paramList.get(Albany::strint("Nonlinear Factor",i), 0.0);

    // Add the factor as a Sacado-ized parameter
    this->registerSacadoParameter(Albany::strint("Multivariate Quadratic Nonlinear Factor",i), paramLib);
  }
}

template<typename EvalT,typename Traits>
void MVQuadratic<EvalT,Traits>::EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  //Teuchos::ParameterList& paramList = m_source_list->sublist("Multivariate Quadratic");
  Teuchos::RCP<PHX::DataLayout> dl = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Source Name"), dl);
  m_source = f ;
  source.addEvaluatedField(m_source);
}
template<typename EvalT,typename Traits>
void MVQuadratic<EvalT,Traits>::DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  //Teuchos::ParameterList& paramList = m_source_list->sublist("Multivariate Quadratic");
  Teuchos::RCP<PHX::DataLayout> dl = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Variable Name"), dl);
  m_baseField = f;
  source.addDependentField(m_baseField.fieldTag());
}
template<typename EvalT,typename Traits>
void MVQuadratic<EvalT,Traits>::FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, PHX::FieldManager<Traits>& fm){
  utils.setFieldData(m_source,   fm);
  utils.setFieldData(m_baseField,fm);
  typename std::vector< typename PHX::template MDField<ScalarT,Cell,Node>::size_type > dims;
  m_source.dimensions(dims);
  m_num_qp = dims[1];
}

template<typename EvalT,typename Traits>
void MVQuadratic<EvalT,Traits>::evaluateFields(typename Traits::EvalData workset){

  ScalarT a = 0.0;
  for (unsigned int j=0; j<m_factor.size(); j++) {
    a += m_factor[j];
  }
  a /= static_cast<double>(m_factor.size());

  // Loop over cells, quad points: compute MVQuadratic Source Term
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t iqp=0; iqp<m_num_qp; iqp++)
      m_source(cell, iqp) = a * m_baseField(cell,iqp) * m_baseField(cell,iqp);
  }
}

template<typename EvalT,typename Traits>
typename MVQuadratic<EvalT,Traits>::ScalarT& 
MVQuadratic<EvalT,Traits>::getValue(const std::string &n)
{
  for (unsigned int i=0; i<m_factor.size(); i++) {
    if (n == Albany::strint("Multivariate Quadratic Nonlinear Factor",i))
      return m_factor[i];
  }
  return m_factor[0];
}

template<typename EvalT, typename Traits>
class MVExponential : public Source_Base<EvalT,Traits>, public Sacado::ParameterAccessor<EvalT, SPL_Traits> {
public :
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  static bool check_for_existance(Teuchos::ParameterList* source_list);
  MVExponential(Teuchos::ParameterList& p);
  virtual ~MVExponential(){}
  virtual void EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p);
  virtual void DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p);
  virtual void FieldData      (PHX::EvaluatorUtilities<EvalT,Traits> &utils, PHX::FieldManager<Traits>& fm);
  virtual void evaluateFields (typename Traits::EvalData workset);
  virtual ScalarT & getValue(const std::string &n);
private :
  std::vector<ScalarT>     m_factor;
  std::size_t m_num_qp;
  Teuchos::ParameterList* m_source_list;
  PHX::MDField<ScalarT,Cell,Point>   m_source;
  PHX::MDField<ScalarT,Cell,Point>   m_baseField;
};

template<typename EvalT,typename Traits>
bool MVExponential<EvalT,Traits>::check_for_existance(Teuchos::ParameterList* source_list)
{
  const bool exists = source_list->getEntryPtr("Multivariate Exponential");
  return exists;
}

template<typename EvalT,typename Traits>
MVExponential<EvalT,Traits>::MVExponential(Teuchos::ParameterList& p) {
  m_source_list = p.get<Teuchos::ParameterList*>("Parameter List", NULL);
  Teuchos::ParameterList& paramList = m_source_list->sublist("Multivariate Exponential");
  int num_vars = paramList.get("Dimension", 1);
  m_factor.resize(num_vars);
  Teuchos::RCP<ParamLib> paramLib = p.get< Teuchos::RCP<ParamLib> > ("Parameter Library", Teuchos::null);
  for (int i=0; i<num_vars; i++) {
    m_factor[i] = paramList.get(Albany::strint("Nonlinear Factor",i), 0.0);

    // Add the factor as a Sacado-ized parameter
    this->registerSacadoParameter(Albany::strint("Multivariate Exponential Nonlinear Factor",i), paramLib);
  }
}

template<typename EvalT,typename Traits>
void MVExponential<EvalT,Traits>::EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  //Teuchos::ParameterList& paramList = m_source_list->sublist("Multivariate Exponential");
  Teuchos::RCP<PHX::DataLayout> dl = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Source Name"), dl);
  m_source = f ;
  source.addEvaluatedField(m_source);
}
template<typename EvalT,typename Traits>
void MVExponential<EvalT,Traits>::DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p) {
  //Teuchos::ParameterList& paramList = m_source_list->sublist("Multivariate Exponential");
  Teuchos::RCP<PHX::DataLayout> dl = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f(p.get<std::string>("Variable Name"), dl);
  m_baseField = f;
  source.addDependentField(m_baseField.fieldTag());
}
template<typename EvalT,typename Traits>
void MVExponential<EvalT,Traits>::FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, PHX::FieldManager<Traits>& fm){
  utils.setFieldData(m_source,   fm);
  utils.setFieldData(m_baseField,fm);
  typename std::vector< typename PHX::template MDField<ScalarT,Cell,Node>::size_type > dims;
  m_source.dimensions(dims);
  m_num_qp = dims[1];
}

template<typename EvalT,typename Traits>
void MVExponential<EvalT,Traits>::evaluateFields(typename Traits::EvalData workset){

  ScalarT a = 0.0;
  for (unsigned int j=0; j<m_factor.size(); j++) {
    a += m_factor[j];
  }
  a /= static_cast<double>(m_factor.size());

  // Loop over cells, quad points: compute MVExponential Source Term
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t iqp=0; iqp<m_num_qp; iqp++)
      m_source(cell, iqp) = a*std::exp(m_baseField(cell,iqp));
  }
}

template<typename EvalT,typename Traits>
typename MVExponential<EvalT,Traits>::ScalarT& 
MVExponential<EvalT,Traits>::getValue(const std::string &n)
{
  for (unsigned int i=0; i<m_factor.size(); i++) {
    if (n == Albany::strint("Multivariate Exponential Nonlinear Factor",i))
      return m_factor[i];
  }
  return m_factor[0];
}


// This needs to be templated to get mesh derivatives
template<typename EvalT>
class Spatial_Base {
public :
  virtual ~Spatial_Base(){}
  typedef typename EvalT::MeshScalarT MeshScalarT;
  virtual MeshScalarT evaluateFields  (const std::vector<MeshScalarT> &coords)=0;
};

template<typename EvalT>
class Gaussian : public Spatial_Base<EvalT> {
public :
  static bool check_for_existance(Teuchos::ParameterList &source_list);
  typedef typename EvalT::MeshScalarT MeshScalarT;
  Gaussian(Teuchos::ParameterList &source_list, std::size_t num);
  virtual ~Gaussian(){}
  virtual MeshScalarT evaluateFields(const std::vector<MeshScalarT> &coords);
private :
  double      m_amplitude        ;
  double      m_radius           ;
  double      m_sigma_sq         ;
  double      m_sigma_pi         ;
  Teuchos::Array<double> m_centroid ;
};

template<typename EvalT>
inline bool Gaussian<EvalT>::check_for_existance(Teuchos::ParameterList &source_list)
{
  std::string g("Gaussian");
  bool exists = source_list.getEntryPtr("Type");
  if (exists) exists = g==source_list.get("Type",g);
  return exists;
}

template<typename EvalT>
inline Gaussian<EvalT>::Gaussian(Teuchos::ParameterList &source_list, std::size_t num)
{
  Teuchos::ParameterList& paramList = source_list.sublist("Spatial",true);
  m_amplitude = paramList.get("Amplitude",      1.0);
  m_radius    = paramList.get("Radius",         1.0);
  m_centroid  = source_list.get(Albany::strint("Center",num),m_centroid);
  m_sigma_sq = 1.0/(2.0*std::pow(m_radius, 2));
  const double pi = 3.1415926535897932385;
  m_sigma_pi = 1.0/(m_radius*std::sqrt(2*pi));
}

template<typename EvalT>
typename EvalT::MeshScalarT Gaussian<EvalT>::
evaluateFields(const std::vector<typename EvalT::MeshScalarT> &coords)
{
  MeshScalarT exponent=0;
  const std::size_t nsd = coords.size();
  for (std::size_t i=0; i<nsd; ++i) {
    exponent += std::pow(m_centroid[i]-coords[i],2);
  }
  exponent *= m_sigma_sq;  
  MeshScalarT x(0.0);
  if (nsd==1)
    x = m_amplitude *          m_sigma_pi    * std::exp(-exponent);           
  else if (nsd==2)
    x = m_amplitude * std::pow(m_sigma_pi,2) * std::exp(-exponent);           
  else if (nsd==3)
    x = m_amplitude * std::pow(m_sigma_pi,3) * std::exp(-exponent);           
  return x;
}


class Wavelet_Base {
public :
  virtual ~Wavelet_Base(){}
  virtual RealType evaluateFields  (const RealType time) = 0;
};

class Monotone : public Wavelet_Base {
public :
  static bool check_for_existance(Teuchos::ParameterList &source_list);
  Monotone(Teuchos::ParameterList &source_list);
  virtual ~Monotone(){}
  virtual RealType evaluateFields  (const RealType time);
private :
};

inline bool Monotone::check_for_existance(Teuchos::ParameterList &source_list)
{ 
  std::string g("Monotone");
  bool exists = source_list.getEntryPtr("Type");
  if (exists) exists = g==source_list.get("Type",g);
  return exists;
}

inline Monotone::Monotone(Teuchos::ParameterList &source_list)
{
}

inline RealType Monotone::evaluateFields(const RealType time)
{ return 1.0; }


template<typename EvalT, typename Traits>
class PointSource : public Source_Base<EvalT,Traits> {
public :
  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;
  static bool check_for_existance(Teuchos::ParameterList* source_list);
  PointSource(Teuchos::ParameterList* source_list);
  virtual ~PointSource();
  virtual void EvaluatedFields (Source<EvalT,Traits> &source, Teuchos::ParameterList& p);
  virtual void DependentFields (Source<EvalT,Traits> &source, Teuchos::ParameterList& p);
  virtual void FieldData       (PHX::EvaluatorUtilities<EvalT,Traits> &utils, PHX::FieldManager<Traits>& fm);
  virtual void evaluateFields  (typename Traits::EvalData workset);
private :
  std::size_t                 m_num_qp;
  std::size_t                 m_num_dim;
  Wavelet_Base               *m_wavelet;
  std::vector<Spatial_Base<EvalT> * >  m_spatials;

  Teuchos::ParameterList* m_source_list;
              PHX::MDField<ScalarT,Cell,Point>   m_pressure_source;
  PHX::MDField<MeshScalarT,Cell,Point,Dim> coordVec;
};
template<typename EvalT, typename Traits>
bool PointSource<EvalT,Traits>::check_for_existance(Teuchos::ParameterList* source_list)
{
  const bool exists = source_list->getEntryPtr("Point");
  return exists;
}

template<typename EvalT, typename Traits>
PointSource<EvalT,Traits>::PointSource(Teuchos::ParameterList* source_list) :
  m_num_qp(0), m_wavelet(NULL), m_spatials(), m_source_list(source_list)
{
  Teuchos::ParameterList& paramList = source_list->sublist("Point",true);
  const std::size_t num  = paramList.get("Number", 0);
  Teuchos::ParameterList& spatial_param = paramList.sublist("Spatial",true);
  if (Gaussian<EvalT>::check_for_existance(spatial_param)) {
    for (std::size_t i=0; i<num; ++i) {
      Gaussian<EvalT> *s = new Gaussian<EvalT>(paramList,i);
      m_spatials.push_back(s);
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(m_spatials.empty(), std::logic_error,
                       "Point: Did not find a single spatial component.  Specify Gaussian.");
  }
  Teuchos::ParameterList& wavelet_param = paramList.sublist("Time Wavelet",true);
  if (Monotone::check_for_existance(wavelet_param)) {
    Monotone *monotone = new Monotone(wavelet_param);
    m_wavelet = monotone;
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(!m_wavelet, std::logic_error,
                       "Point: Did not find a single wavelet component. Specify Monotone.");
  }
}
template<typename EvalT, typename Traits>
PointSource<EvalT,Traits>::~PointSource()
{
  delete m_wavelet;
  for (std::size_t i=0; i<m_spatials.size(); ++i) {
    Spatial_Base<EvalT> *s = m_spatials[i];
    delete s;
  }
  m_spatials.clear();
}

template<typename EvalT, typename Traits>
void PointSource<EvalT,Traits>::EvaluatedFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p)
{
  Teuchos::RCP<PHX::DataLayout> scalar_qp = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  PHX::MDField<ScalarT,Cell,Point> f0(p.get<std::string>("Pressure Source Name"), scalar_qp);
  m_pressure_source       = f0;
  source.addEvaluatedField(m_pressure_source);
}

template<typename EvalT, typename Traits>
void PointSource<EvalT,Traits>::DependentFields(Source<EvalT,Traits> &source, Teuchos::ParameterList& p)
{
  Teuchos::RCP<PHX::DataLayout> scalar_qp = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Scalar Data Layout");
  Teuchos::RCP<PHX::DataLayout> vector_qp = p.get< Teuchos::RCP<PHX::DataLayout> >("QP Vector Data Layout");

  PHX::MDField<MeshScalarT,Cell,Point,Dim> f0
    (p.get<std::string>("QP Coordinate Vector Name"),  vector_qp);
  coordVec = f0;
  source.addDependentField(coordVec.fieldTag());
}

template<typename EvalT, typename Traits>
void    PointSource<EvalT,Traits>::FieldData(PHX::EvaluatorUtilities<EvalT,Traits> &utils, PHX::FieldManager<Traits>& fm){
  utils.setFieldData(m_pressure_source, fm);
  utils.setFieldData(coordVec,     fm);
  typename std::vector< typename PHX::template MDField<ScalarT,Cell,Point,Dim>::size_type > dims;
  coordVec.dimensions(dims);
  m_num_qp  = dims[1];
  m_num_dim = dims[2];
}

template<typename EvalT, typename Traits>
void PointSource<EvalT,Traits>::evaluateFields(typename Traits::EvalData workset)
{
  const RealType time  = workset.current_time;
  for (std::size_t cell = 0; cell < workset.numCells; ++cell) {
    for (std::size_t iqp=0; iqp<m_num_qp; iqp++) {
      std::vector<MeshScalarT> coord;
      for (std::size_t i=0; i<m_num_dim; ++i) {
        const MeshScalarT  x  = coordVec(cell,iqp,i);
        coord.push_back(x);
      }
      //ScalarT &p_s = m_pressure_source(cell,iqp);
      m_pressure_source(cell,iqp) = 0;
      const RealType wavelet = m_wavelet->evaluateFields(time);
      for (std::size_t i=0; i<m_spatials.size(); ++i) {
        const MeshScalarT spatial = m_spatials[i]->evaluateFields(coord);
        m_pressure_source(cell,iqp) += wavelet*spatial;
      }
    }
  }
}


}

using namespace Source_Functions;

template<typename EvalT, typename Traits>
Source<EvalT, Traits>::Source(Teuchos::ParameterList& p)
{

  Teuchos::ParameterList* source_list = p.get<Teuchos::ParameterList*>("Parameter List", NULL);
  if (Constant<EvalT,Traits>::check_for_existance(source_list)) {
    Constant<EvalT,Traits>    *q = new Constant<EvalT,Traits>(p);
    Source_Base<EvalT,Traits> *sb = q;
    m_sources.push_back(sb);
    this->setName("ConstantSource" );
  }
  if (Table<EvalT,Traits>::check_for_existance(source_list)) {
    Table<EvalT,Traits>    *q = new Table<EvalT,Traits>(p);
    Source_Base<EvalT,Traits> *sb = q;
    m_sources.push_back(sb);
    this->setName("TableSource" );
  }
  if (Trigonometric<EvalT,Traits>::check_for_existance(source_list)) {
    Trigonometric<EvalT,Traits>    *q = new Trigonometric<EvalT,Traits>(p);
    Source_Base<EvalT,Traits> *sb = q;
    m_sources.push_back(sb);
    this->setName("TrigonometricSource" );
  }
#ifdef ALBANY_STOKHOS
  if (TruncatedKL<EvalT,Traits>::check_for_existance(source_list)) {
    TruncatedKL<EvalT,Traits>    *q = new TruncatedKL<EvalT,Traits>(p);
    Source_Base<EvalT,Traits> *sb = q;
    m_sources.push_back(sb);
    this->setName("TruncatedKLSource" );
  }
#endif
  if (Quadratic<EvalT,Traits>::check_for_existance(source_list)) {
    Quadratic<EvalT,Traits>    *q = new Quadratic<EvalT,Traits>(p);
    Source_Base<EvalT,Traits> *sb = q;
    m_sources.push_back(sb);
    this->setName("QuadraticSource" );
  }
  if (NeutronFission<EvalT,Traits>::check_for_existance(source_list)) {
    NeutronFission<EvalT,Traits>    *q = new NeutronFission<EvalT,Traits>(p);
    Source_Base<EvalT,Traits> *sb = q;
    m_sources.push_back(sb);
    this->setName("NeutronFissionSource" );
  }
  if (MVQuadratic<EvalT,Traits>::check_for_existance(source_list)) {
    MVQuadratic<EvalT,Traits> *q = new MVQuadratic<EvalT,Traits>(p);
    Source_Base<EvalT,Traits> *sb = q;
    m_sources.push_back(sb);
    this->setName("MVQuadraticSource" );
  }
  if (MVExponential<EvalT,Traits>::check_for_existance(source_list)) {
    MVExponential<EvalT,Traits> *q = new MVExponential<EvalT,Traits>(p);
    Source_Base<EvalT,Traits> *sb = q;
    m_sources.push_back(sb);
    this->setName("MVExponentialSource" );
  }
  if (PointSource<EvalT,Traits>::check_for_existance(source_list)) {
    PointSource<EvalT,Traits>       *s = new PointSource<EvalT,Traits>(source_list);
    Source_Base<EvalT,Traits> *sb = s;
    m_sources.push_back(sb);
    this->setName("PointSource" );
  }
  for (std::size_t i=0; i<m_sources.size(); ++i) {
    Source_Base<EvalT,Traits>* sb =  m_sources[i];
    sb->DependentFields(*this, p);
    sb->EvaluatedFields(*this, p);
  }

}

template<typename EvalT, typename Traits>
Source<EvalT, Traits>::~Source() {
  for (std::size_t i=0; i<m_sources.size(); ++i) {
    Source_Base<EvalT,Traits>* sb =  m_sources[i];
    delete sb;
  }
  m_sources.clear();
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Source<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  for (std::size_t i=0; i<m_sources.size(); ++i) {
    Source_Base<EvalT,Traits>* sb =  m_sources[i];
    sb->FieldData(this->utils, fm);
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
void Source<EvalT, Traits>::evaluateFields(typename Traits::EvalData workset)
{
  for (std::size_t i=0; i<m_sources.size(); ++i) {
    Source_Base<EvalT,Traits>* sb =  m_sources[i];
    sb->evaluateFields(workset);
  }
  return;
}

//**********************************************************************
}

