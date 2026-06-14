#include "seki_hw/SekiHwTypes.h"
#include "seki_hw/SekiHwOps.h"
#include "seki_hw/SekiHwAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "seki_hw/SekiHwOps.cpp.inc"

namespace mlir::seki_hw {

LogicalResult MatmulOp::verify() {
    auto lhsElem = cast<TileType>(getLhs().getType()).getElementType();
    auto rhsElem = cast<TileType>(getRhs().getType()).getElementType();
    auto accElem = cast<TileType>(getAcc().getType()).getElementType();
    auto resElem = cast<TileType>(getResult().getType()).getElementType();
    if (accElem != resElem)
        return emitOpError("acc and result must have the same element type");
    if (lhsElem != rhsElem)
        return emitOpError("lhs and rhs must have the same element type");
    bool valid = (lhsElem.isF32()     && accElem.isF32()) ||
                 (lhsElem.isBF16()    && accElem.isF32()) ||
                 (lhsElem.isInteger(8)&& accElem.isInteger(32));
    if (!valid)
        return emitOpError("invalid (lhs, acc) element type pair; "
                           "valid: (f32, f32), (bf16, f32), (i8, i32)");

    return success();
}

LogicalResult TileSplatOp::verify() {
    auto resultElem = cast<TileType>(getResult().getType()).getElementType();
    if (getScalar().getType() != resultElem)
        return emitOpError("scalar type must match tile element type");
    return success();
}

LogicalResult TileReduceVecOp::verify() {
    auto srcElem = cast<TileType>(getSrc().getType()).getElementType();
    auto resType = cast<VecType>(getResult().getType());
    if (srcElem != resType.getElementType())
        return emitOpError("src tile and result vec must have the same element type");
    return success();
}

LogicalResult VecBroadcastTileOp::verify() {
    auto vecType = cast<VecType>(getVec().getType());
    auto resElem = cast<TileType>(getResult().getType()).getElementType();
    if (vecType.getElementType() != resElem)
        return emitOpError("vec and result tile must have the same element type");
    return success();
}

} // namespace mlir::seki_hw
