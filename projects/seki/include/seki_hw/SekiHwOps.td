#ifndef SEKI_HW_OPS_TD
#define SEKI_HW_OPS_TD

include "seki_hw/SekiHwDialect.td"
include "seki_hw/SekiHwTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SekiHwOps<string mnemonic, list<Trait> traits = []>
    : Op<SekiHwDialect, mnemonic, traits>;

#endif // SEKI_HW_OPS_TD
