// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// Utilities to help passing mock objects through dependency injection.

// These classes allow passing a mock object during tests but still building
// objects in initializer lists when no mock is passed. Use `MakeOwnedIfNoMock`
// and `GetPointerToMockOrOwned` together. The first constructs the non-mock
// object only if no mock is received. The second gets a pointer to the to
// the non-mock object if no mock is received, or gets a pointer to the mock
// otherwise. Together they initialize two constructor fields: the first
// initializes a const unique_ptr to hold the object constructed (if
// constructed) and the second initializes a const pointer with whatever object
// should be used (the constructed object or the mock). For an example, see
// regularied_evolution_search_algorithm.cc.

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_MOCKING_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_MOCKING_H_

#include <memory>

namespace brain {
namespace evolution {

template <class Mockable, class MockableInterface, class... Args>
std::unique_ptr<MockableInterface> MakeOwnedIfNoMock(const bool enabled,
                                                     MockableInterface* mock,
                                                     Args... args) {
  if (!enabled) {
    return std::unique_ptr<MockableInterface>();
  }
  return MakeOwnedIfNoMock<Mockable, MockableInterface, Args...>(mock, args...);
}

template <class Mockable, class MockableInterface, class... Args>
std::unique_ptr<MockableInterface> MakeOwnedIfNoMock(MockableInterface* mock,
                                                     Args... args) {
  if (mock == nullptr) {
    // No mock received. Construct owned object and return it.
    return std::make_unique<Mockable>(args...);
  } else {
    // We will be using the mock. Return a nullptr.
    return std::unique_ptr<MockableInterface>();
  }
}

template <class MockableInterface, class MockableOrMockableInterface>
MockableInterface* GetPointerToMockOrOwned(const bool enabled,
                                           MockableInterface* mock,
                                           MockableOrMockableInterface* owned) {
  if (!enabled) {
    return nullptr;
  }
  return GetPointerToMockOrOwned<MockableInterface,
                                 MockableOrMockableInterface>(mock, owned);
}

template <class MockableInterface, class MockableOrMockableInterface>
MockableInterface* GetPointerToMockOrOwned(MockableInterface* mock,
                                           MockableOrMockableInterface* owned) {
  if (mock == nullptr) {
    CHECK(owned != nullptr);
    return owned;
  } else {
    CHECK(owned == nullptr);
    return mock;
  }
}

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_MOCKING_H_
