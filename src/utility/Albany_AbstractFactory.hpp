#ifndef ALBANY_ABSTRACT_FACTORY_HPP
#define ALBANY_ABSTRACT_FACTORY_HPP

#include <Teuchos_RCP.hpp>
#include <list>

namespace Albany {

// A generic factory.
template<typename AbstractProduct,
         typename KeyType,
         typename... CtorArgs>
class AbstractFactory {
public:

  using obj_type = AbstractProduct;
  using key_type = std::string;
  using obj_ptr_type = Teuchos::RCP<obj_type>;

  virtual ~AbstractFactory () = default;

  virtual obj_ptr_type create (const key_type&, const CtorArgs&&...) const = 0;

  virtual bool provides (const key_type&) const = 0;
};

// A container for factories of the same type.
template<typename AbstractFactoryType>
class FactoriesContainer;

// Handy Specialization, where the ctor args is deduced from factory type
template<typename AbstractProduct,
         typename KeyType,
         typename... CtorArgs>
class FactoriesContainer<AbstractFactory<AbstractProduct,KeyType,CtorArgs...>> {
public:

  using factory_type = AbstractFactory<AbstractProduct,KeyType,CtorArgs...>;
  using key_type     = typename factory_type::key_type;
  using obj_ptr_type = typename factory_type::obj_ptr_type;

  virtual ~FactoriesContainer () = default;

  void add_factory (const factory_type& factory) {
    m_factories.push_back(std::cref(factory));
  }

  obj_ptr_type create (const key_type& key, const CtorArgs&&... args) const {
    // Check that the user did not forget to register factories
    TEUCHOS_TEST_FOR_EXCEPTION (m_factories.empty(), std::runtime_error,
      "Error! No factories registered in the factory container.\n"
      "       Did you forget to call add_factory() for all the factories?\n");

    obj_ptr_type product;

    for (const auto& factory : m_factories) {
      if (factory.get().provides(key)) {
        product = factory.get().create(key,args...);
        break;
      }
    }

    TEUCHOS_TEST_FOR_EXCEPTION (product.is_null(), std::runtime_error,
      "Error! Could not create object with key '" + key + "'.\n"
      "       Did you forget to register the proper factory?\n");

    return product;
  }

  static FactoriesContainer& instance () {
    static FactoriesContainer container;
    return container;
  }

private:

  FactoriesContainer () = default;

  std::list<std::reference_wrapper<const factory_type>> m_factories;
};

} // namespace Albany

#endif // ALBANY_ABSTRACT_FACTORY_HPP
