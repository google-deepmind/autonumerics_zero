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

// Types, conversions, and simple inline checks.

#ifndef LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_TYPES_H_
#define LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_TYPES_H_

#include <sched.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <ostream>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"

namespace brain {
namespace evolution {

// `Int` is the preferred type for all integers. Use this unless there is a
// reason not to. Reasons could be the demands of external interfaces or
// speed/space considerations. Must be castable to `RNGSeed`.
typedef int64_t Integer;  // A generic integer.

// Type for seeding random generators. We use a 32-bit unsigned int because
// that's what `MTRandom` accepts. Must be castable from `Integer`.
typedef uint32_t RNGSeed;

typedef std::atomic_llong AtomicInteger;

// Convenience methods to parse protos.
template <class ProtoT>
ProtoT ParseSerialized(absl::string_view str) {
  ProtoT proto;
  CHECK(proto.ParseFromString(str));
  return proto;
}
template <class ProtoT>
ProtoT ParseTextFormat(absl::string_view str) {
  ProtoT proto;
  CHECK(::google::protobuf::TextFormat::ParseFromString(str, &proto));
  return proto;
}
template <class ProtoT>
std::unique_ptr<ProtoT> TryParseTextFormat(absl::string_view str) {
  auto proto = std::make_unique<ProtoT>();
  if (::google::protobuf::TextFormat::ParseFromString(str, proto.get())) {
    return proto;
  } else {
    return std::unique_ptr<ProtoT>();
  }
}

// Returns the extension of a proto.
template <class ProtoT, class ExtensionT>
ExtensionT ParseTextFormatExtension(absl::string_view proto_str) {
  const auto proto = ParseTextFormat<ProtoT>(proto_str);
  return proto.GetExtension(ExtensionT::ext);
}

// Convenience methods to parse initializer list arguments.
template <typename NumericT>
NumericT PositiveOrDie(const NumericT value) {
  CHECK_GT(value, NumericT()) << "Found non-positive." << std::endl;
  return value;
}
template <typename PointerT>
PointerT NotNullOrDie(PointerT value) {
  CHECK(value != nullptr) << "Found null." << std::endl;
  return value;
}
template <typename ContainerT>  // Also works for strings.
const ContainerT& NonEmptyOrDie(const ContainerT& value) {
  CHECK(!value.empty()) << "Found empty." << std::endl;
  return value;
}
template <typename ContainerT>  // Also works for strings.
ContainerT& NonEmptyOrDie(ContainerT& value) {
  CHECK(!value.empty()) << "Found empty." << std::endl;
  return value;
}
template <typename ContainerT>  // Also works for strings.
ContainerT* NonEmptyOrDie(ContainerT* value) {
  CHECK(!value->empty()) << "Found empty." << std::endl;
  return value;
}
template <typename ContainerT>  // Also works for strings.
const ContainerT& SizeLessThanOrDie(const ContainerT& value,
                                    const size_t max_size) {
  CHECK_LT(value.size(), max_size) << "Too large." << std::endl;
  return value;
}
template <typename ContainerT>  // Also works for strings.
ContainerT& SizeLessThanOrDie(ContainerT& value, const size_t max_size) {
  CHECK_LT(value.size(), max_size) << "Too large." << std::endl;
  return value;
}
template <typename ContainerT>  // Also works for strings.
ContainerT* SizeLessThanOrDie(ContainerT* value, const size_t max_size) {
  CHECK_LT(value->size(), max_size) << "Too large." << std::endl;
  return value;
}

}  // namespace evolution
}  // namespace brain

#endif  // LEARNING_BRAIN_RESEARCH_EVOLUTION_LIB_TYPES_H_
