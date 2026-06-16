#ifndef SEKI_HW_PASSES_H
#define SEKI_HW_PASSES_H

#include "mlir/Pass/Pass.h"

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "seki_hw/SekiHwPasses.h.inc"

#endif // SEKI_HW_PASSES_H
