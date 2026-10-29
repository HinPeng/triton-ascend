// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchLayoutPlan.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <optional>
namespace mlir::triton::cfg {
namespace {
using E = BatchExprRef;
using K = BatchExpr;
using Key = std::pair<K::Kind, int64_t>;
// Deliberately bounded exact integer algebra. No sampled dynamic stride or
// equality of byte strides can establish a logical direction.
struct Linear {
  int64_t constant = 0;
  std::map<Key, int64_t> terms;
};
std::optional<Linear> linear(E e, unsigned &budget) {
  if (!e || !budget)
    return std::nullopt;
  --budget;
  if (e->kind == K::Constant)
    return Linear{e->value, {}};
  if (e->kind == K::Coordinate || e->kind == K::Program ||
      e->kind == K::Argument || e->kind == K::Grid)
    return Linear{0, {{{e->kind, e->value}, 1}}};
  // Signed extension is exact under the existing no-wrap guard. Unsigned
  // extension and truncation require additional range facts: KEEP for now.
  if (e->kind == K::ExtSI)
    return linear(e->lhs, budget);
  if (e->kind != K::Add && e->kind != K::Sub && e->kind != K::Mul)
    return std::nullopt;
  auto a = linear(e->lhs, budget), b = linear(e->rhs, budget);
  if (!a || !b)
    return std::nullopt;
  if (e->kind == K::Mul) {
    if (!a->terms.empty() && !b->terms.empty())
      return std::nullopt;
    if (b->terms.empty())
      std::swap(a, b);
    int64_t scale = a->constant;
    if (llvm::MulOverflow(b->constant, scale, b->constant))
      return std::nullopt;
    for (auto &[key, value] : b->terms)
      if (llvm::MulOverflow(value, scale, value))
        return std::nullopt;
    return b;
  }
  if (e->kind == K::Sub) {
    if (llvm::MulOverflow(b->constant, int64_t(-1), b->constant))
      return std::nullopt;
    for (auto &[key, value] : b->terms)
      if (llvm::MulOverflow(value, int64_t(-1), value))
        return std::nullopt;
  }
  if (llvm::AddOverflow(a->constant, b->constant, a->constant))
    return std::nullopt;
  for (auto [key, value] : b->terms)
    if (llvm::AddOverflow(a->terms[key], value, a->terms[key]))
      return std::nullopt;
  return a;
}
bool contains(E e, K::Kind kind, int64_t id = -1) {
  if (!e)
    return false;
  return (e->kind == kind && (id < 0 || e->value == id)) ||
         contains(e->lhs, kind, id) || contains(e->rhs, kind, id);
}
void conjuncts(E mask, SmallVectorImpl<E> &out) {
  if (!mask)
    return;
  if (mask->kind == K::And) {
    conjuncts(mask->lhs, out);
    conjuncts(mask->rhs, out);
  } else
    out.push_back(mask);
}
void coordinateIDs(E e, SmallVectorImpl<int64_t> &ids) {
  if (!e)
    return;
  if (e->kind == K::Coordinate && !llvm::is_contained(ids, e->value))
    ids.push_back(e->value);
  coordinateIDs(e->lhs, ids);
  coordinateIDs(e->rhs, ids);
}
bool rangeGroup(E e, ArrayRef<int64_t> axes) {
  if (!e)
    return true;
  if (e->rangeOrigin) {
    SmallVector<int64_t> ids;
    coordinateIDs(e, ids);
    llvm::sort(ids);
    return ArrayRef<int64_t>(ids) == axes;
  }
  if (e->kind == K::Coordinate)
    return false;
  return rangeGroup(e->lhs, axes) && rangeGroup(e->rhs, axes);
}
// A bound on a coordinate, before byte scaling, associates that coordinate's
// local projections with the selected program direction. It excludes all
// other program directions, even if their strides happen to be equal.
bool bind(E predicate, const BatchAccessFact &fact, BatchLayoutPlan &plan) {
  if (predicate->kind != K::SLT || contains(predicate->rhs, K::Coordinate) ||
      contains(predicate->rhs, K::Program) ||
      contains(predicate->rhs, K::Induction))
    return false;
  unsigned budget = 256;
  auto form = linear(predicate->lhs, budget);
  if (!form)
    return false;
  SmallVector<int64_t> axes;
  for (auto [key, value] : form->terms) {
    if (!value)
      continue;
    if (key.first == K::Coordinate && value > 0)
      axes.push_back(key.second);
    else if (key != Key{K::Program, plan.gridAxis})
      return false;
  }
  if (axes.empty())
    return false;
  int64_t extent = 1;
  for (int64_t axis : llvm::reverse(axes)) {
    if (axis < 0 || axis >= static_cast<int64_t>(fact.shape.size()) ||
        form->terms[{K::Coordinate, axis}] != extent || fact.shape[axis] <= 0 ||
        llvm::MulOverflow(extent, fact.shape[axis], extent))
      return false;
  }
  if (form->terms[{K::Program, plan.gridAxis}] != extent)
    return false;
  plan.localAxes = axes;
  plan.tileExtent = extent;
  plan.origin = form->constant;
  plan.coordinate = predicate->lhs;
  plan.upper = predicate->rhs;
  return true;
}
} // namespace
SmallVector<BatchLayoutPlan, 0>
planBatchLayouts(const BatchAccessReport &accesses, int axis, int64_t factor) {
  SmallVector<BatchLayoutPlan, 0> plans;
  for (auto [id, fact] : llvm::enumerate(accesses.accesses)) {
    BatchLayoutPlan plan;
    plan.accessID = id;
    plan.gridAxis = axis;
    plan.shape = fact.shape;
    llvm::raw_string_ostream stream(plan.location);
    fact.operation->getLoc().print(stream);
    SmallVector<E> predicates;
    conjuncts(fact.mask, predicates);
    bool bound = false, unknown = false;
    for (auto predicate : predicates) {
      if (predicate->kind == K::Unknown)
        unknown = true;
      if (contains(predicate, K::Program, axis)) {
        if (bound || !bind(predicate, fact, plan)) {
          unknown = true;
          continue;
        }
        bound = true;
      }
    }
    if (!contains(fact.byteOffset, K::Program, axis)) {
      plan.reason = "SHARED_ACCESS";
      plan.maskClass = "shared";
    } else if (!bound && !unknown && !contains(fact.mask, K::Program, axis)) {
      plan.reason = "ALREADY_STRUCTURED";
      plan.maskClass = "independent_axes";
    } else if (bound && !unknown) {
      // Direction is established above. Check the byte-stride chain only now.
      unsigned budget = 512;
      auto address = linear(fact.byteOffset, budget);
      bool strides = bool(address);
      if (address) {
        int64_t weight = 1;
        int64_t stride = address->terms[{K::Coordinate, plan.localAxes.back()}];
        strides = stride > 0;
        for (int64_t local : llvm::reverse(plan.localAxes)) {
          int64_t expected;
          if (llvm::MulOverflow(weight, stride, expected) ||
              address->terms[{K::Coordinate, local}] != expected)
            strides = false;
          if (llvm::MulOverflow(weight, fact.shape[local], weight))
            strides = false;
        }
        int64_t expected;
        if (llvm::MulOverflow(weight, stride, expected) ||
            address->terms[{K::Program, axis}] != expected)
          strides = false;
      }
      bool extraMask = false;
      for (auto predicate : predicates) {
        if (predicate->lhs == plan.coordinate && predicate->rhs == plan.upper)
          continue;
        for (int64_t local : plan.localAxes)
          if (contains(predicate, K::Coordinate, local))
            extraMask = true;
      }
      int64_t total;
      if (llvm::MulOverflow(plan.tileExtent, factor, total) ||
          total > (1 << 20))
        plan.reason = "ANALYSIS_BUDGET_EXCEEDED";
      else if (plan.localAxes.size() > 1 &&
               !rangeGroup(plan.coordinate, plan.localAxes))
        plan.reason = "MULTI_AXIS_PROVENANCE_UNPROVEN";
      else if (!strides)
        plan.reason = "STRIDE_CHAIN_UNPROVEN";
      else if (extraMask)
        plan.reason = "MASK_NOT_INTERVAL_OR_RECTANGLE";
      else {
        plan.action = plan.localAxes.front() == 0
                          ? BatchLayoutPlan::MemoryView
                          : BatchLayoutPlan::PointwiseRegion;
        plan.maskClass = "linear_prefix";
        plan.reason = "PROVEN_DIRECTION_AND_STRIDE_CHAIN";
        // Memory-only collapse is free of permutations only for a leading
        // group. Nonadjacent groups need a proved pointwise region/cost plan.
        bool leading = true;
        for (auto [position, local] : llvm::enumerate(plan.localAxes))
          leading &= local == static_cast<int64_t>(position);
        bool columnRegion =
            fact.shape.size() == 2 && plan.localAxes == SmallVector<int64_t>{1};
        bool simple = (leading || columnRegion) && fact.shape.size() <= 3 &&
                      fact.loops.empty();
        simple &= address->constant % fact.bytes == 0;
        int64_t stride = address->terms[{K::Coordinate, plan.localAxes.back()}];
        simple &= stride % fact.bytes == 0;
        plan.elementOrigin = address->constant / fact.bytes;
        plan.elementStride = stride / fact.bytes;
        __int128 maxCoordinate =
            (__int128(INT32_MAX) + factor) * plan.tileExtent;
        __int128 magnitude = maxCoordinate * plan.elementStride;
        magnitude += plan.elementOrigin < 0 ? -__int128(plan.elementOrigin)
                                            : plan.elementOrigin;
        for (auto [key, coefficient] : address->terms) {
          if (!coefficient || key == Key{K::Program, axis} ||
              (key.first == K::Coordinate &&
               llvm::is_contained(plan.localAxes, key.second)))
            continue;
          if (coefficient % fact.bytes)
            simple = false;
          // The first row emitter supports original program/coordinate axes.
          // Dynamic address arguments require symbolic-stride planning.
          if (key.first != K::Coordinate && key.first != K::Program &&
              key.first != K::Grid)
            simple = false;
          int64_t c = coefficient / fact.bytes;
          plan.restTerms.push_back({key.first, key.second, c});
          magnitude += (__int128(INT32_MAX) + 1) * (c < 0 ? -__int128(c) : c);
        }
        for (auto predicate : predicates)
          if (predicate->lhs != plan.coordinate || predicate->rhs != plan.upper)
            plan.residualMasks.push_back(predicate);
        __int128 maxIndex = maxCoordinate + plan.origin;
        bool endpoints = magnitude * fact.bytes <= INT64_MAX &&
                         maxIndex <= INT64_MAX && maxIndex >= INT64_MIN;
        simple &= endpoints;
        if (!endpoints) {
          plan.action = BatchLayoutPlan::Keep;
          plan.reason = "NEW_BYTE_ENDPOINT_UNPROVEN";
        }
        plan.linearMemoryView = simple;
      }
    } else if (unknown)
      plan.reason = "MASK_NOT_INTERVAL_OR_RECTANGLE";
    plans.push_back(std::move(plan));
  }
  return plans;
}
std::string serializeBatchLayouts(ArrayRef<BatchLayoutPlan> plans) {
  llvm::json::Array rows;
  for (const auto &plan : plans) {
    llvm::json::Array axes, shape;
    for (auto axis : plan.localAxes)
      axes.push_back(axis);
    for (auto extent : plan.shape)
      shape.push_back(extent);
    rows.push_back(llvm::json::Object{
        {"access_id", plan.accessID},
        {"action", plan.action == BatchLayoutPlan::Keep ? "Keep"
                   : plan.action == BatchLayoutPlan::MemoryView
                       ? "MemoryView"
                       : "PointwiseRegion"},
        {"selected_grid_axis", plan.gridAxis},
        {"local_axes", std::move(axes)},
        {"original_shape", std::move(shape)},
        {"tile_extent", plan.tileExtent},
        {"origin", plan.origin},
        {"reason", plan.reason},
        {"mask_class", plan.maskClass},
        {"source_location", plan.location},
        {"rewritten", plan.rewritten},
        {"bridge", !plan.rewritten ? "not_applied"
                   : plan.action == BatchLayoutPlan::PointwiseRegion
                       ? "direct_pointwise"
                       : "ordered_view"},
        {"expected_memory_form",
         plan.rewritten ? "strided_block_candidate" : "semantic"},
        {"proof_scope", "existing_guard_no_wrap"}});
  }
  return llvm::formatv("{0}",
                       llvm::json::Value(llvm::json::Object{
                           {"planner_version", kBatchLayoutPlannerVersion},
                           {"accesses", std::move(rows)}}))
      .str();
}
} // namespace mlir::triton::cfg
