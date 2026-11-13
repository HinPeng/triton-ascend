// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// Compile and link only the owned facts and production codec, without MLIR.
#include "TritonToGraph/BlockBatching/BatchFactsJSON.h"
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <utility>

using namespace mlir::triton::cfg;

template <typename T, typename = void> struct HasOperation : std::false_type {};
template <typename T>
struct HasOperation<T, std::void_t<decltype(std::declval<T>().operation)>>
    : std::true_type {};
static_assert(!HasOperation<BatchAccessFact>::value,
              "owned access facts must not retain an IR operation");

static void check(bool ok, const char *message) {
  if (!ok) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

int main() {
  BatchEffectProjection projection;
  {
    BatchAccessFact source;
    source.root = 2;
    source.bytes = 4;
    source.write = true;
    source.shape = {8};
    source.byteOffset = BatchExpr::leaf(BatchExpr::Argument, 3, 64);
    source.pointerChecks = {source.byteOffset};
    // The wire order is innermost first, not numeric LoopId order.
    BatchLoopFact inner, outer;
    inner.id = 7;
    outer.id = 2;
    inner.bits = outer.bits = 32;
    inner.lower = outer.lower = BatchExpr::leaf(BatchExpr::Constant, 0, 32);
    inner.upper = outer.upper = BatchExpr::leaf(BatchExpr::Argument, 4, 32);
    inner.step = outer.step = BatchExpr::leaf(BatchExpr::Constant, 1, 32);
    source.loops = {inner, outer};
    projection.accesses = {source, source};
    projection.accesses[1].mask = BatchExpr::leaf(BatchExpr::False);
    check(!source.mask, "effect normalization must not overwrite source facts");
  }
  // The producer's objects have died; expressions and slots remain owned.
  auto wire = batchAccessesJSON(projection);
  auto *object = wire.getAsObject();
  auto *accesses = object->getArray("accesses");
  check(object->getInteger("version") == 2 && accesses->size() == 2,
        "legacy access protocol and member slots must remain intact");
  auto *covered = (*accesses)[1].getAsObject();
  check(covered->getInteger("root") == 2 &&
            covered->getBoolean("write") == true,
        "covered member must preserve its effect identity");
  check(covered->getArray("mask")->front().getAsString() == "false",
        "covered member suppresses only its redundant alias hull");
  check(covered->getArray("pointer_checks")->size() == 1,
        "covered member must still retain original pointer arithmetic");
  auto *loops = covered->getArray("loops");
  check((*loops)[0].getAsObject()->getInteger("id") == 7 &&
            (*loops)[1].getAsObject()->getInteger("id") == 2,
        "inner-to-outer loop order must not change");
  check(!(*loops)[0].getAsObject()->get("carriesValues"),
        "compiler-only facts must not leak into the runtime ABI");
  std::cout << "owned effect projection regressions passed\n";
}
