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

#include "evolution/lib/current_time.h"

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "evolution/lib/types.h"

namespace brain {
namespace evolution {

using ::absl::GetCurrentTimeNanos;
using ::absl::Time;

CurrentTime::CurrentTime() {}

Integer CurrentTime::Nanos() const { return GetCurrentTimeNanos(); }

Time CurrentTime::Now() const { return absl::Now(); }

}  // namespace evolution
}  // namespace brain
