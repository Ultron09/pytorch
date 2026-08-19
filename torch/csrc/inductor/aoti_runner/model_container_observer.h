#pragma once

#include <cstddef>
#include <cstdint>

namespace torch::inductor {

// Lifecycle events an AOTInductor model container reports to an observer.
enum class AOTIContainerEvent {
  // kCreate and kLoadConstants bracket work that happens before/around the
  // container handle itself, so AOTIModelContainerRunner cannot fire them.
  // They are reserved for the driver that owns container construction (e.g.
  // FbAOTInductorModel); observers attached to a bare runner will not see them.
  kCreate, // container construction / model .so load
  kLoadConstants, // constants/weights loaded into a buffer
  kUpdateConstantBuffer, // in-place constants/weights update
  kSwapConstantBuffer, // active <-> inactive constants-buffer swap
  kRunConstantFolding, // constant-folding pass
  // run() / boxed_run(); warmup is inference too. Single-threaded mode is a
  // construction-time flag that swaps run_func_ inside these, not a separate
  // entry point.
  kInference,
  kFreeInactiveBuffer, // release of the inactive constants buffer
};

// Context passed with each event. Fields that don't apply to a given event keep
// their defaults. Deliberately POD (no aten/c10 dependency) so it stays cheap
// and safe to construct on the hot path.
struct AOTIObserverContext {
  const char* device_str = nullptr; // e.g. "cuda", "cpu"
  int device_index = -1;
  size_t num_constants = 0;
  int64_t num_bytes = 0; // e.g. constant bytes moved, when known
  bool use_inactive = false; // update/fold target buffer
};

// Observer for AOTInductor container lifecycle events. Attach a subclass to a
// container runner/driver to receive begin/end callbacks bracketing each event;
// measure durations yourself between onBegin and onEnd. Attaching is optional
// -- a null observer is zero overhead. Implementations must be cheap and must
// not throw (callbacks may run on the serving hot path and during teardown,
// where an escaping exception would terminate the process).
class AOTIModelContainerObserver {
 public:
  virtual ~AOTIModelContainerObserver() = default;

  virtual void onBegin(
      AOTIContainerEvent event,
      const AOTIObserverContext& ctx) = 0;
  virtual void onEnd(
      AOTIContainerEvent event,
      const AOTIObserverContext& ctx) = 0;
};

} // namespace torch::inductor
