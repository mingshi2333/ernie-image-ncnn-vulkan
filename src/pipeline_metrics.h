// SPDX-License-Identifier: MIT
#pragma once
#include "execution_metrics.h"
#include <ernie/pipeline.h>
namespace ernie {
// Private diagnostic entry point. The public API remains unchanged and does
// not collect timing data unless the CLI explicitly supplies a collector.
GenerationResult generate_with_metrics(const GenerationRequest&, const ProgressCallback&, ExecutionMetrics&);
}
