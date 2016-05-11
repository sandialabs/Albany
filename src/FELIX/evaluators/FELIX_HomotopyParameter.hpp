#ifndef FELIX_HOMOTOPY_PARAMETER_HPP
#define FELIX_HOMOTOPY_PARAMETER_HPP 1

namespace FELIX
{

template<typename EvalT>
class HomotopyParameter
{
public:
    static typename EvalT::ScalarT value;
};

template<typename EvalT>
typename EvalT::ScalarT HomotopyParameter<EvalT>::value = 0;

} // Namespace FELIX

#endif // FELIX_HOMOTOPY_PARAMETER_HPP
