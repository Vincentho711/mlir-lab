#ifndef SEKI_PASSES_TD
#define SEKI_PASSES_TD

include "mlir/Pass/PassBase.td"

def SekiAttachTargetPass : Pass<"seki-attach-target", "::mlir::ModuleOp"> {
    let summary = "Resolve --seki-target and attach #seki.target<...> to the module";
    let description = [{
        Reads the seki-target option, resolves it as either a named built-in target or an
        external .mlir attribute file, and attaches the resolved #seki.target<...> attribute
        to the module for use by downstream passes.
    }];
    let options = [
        Option<"target", "seki-target", "std::string", [{""}],
               "Named target (e.g. seki-v1) or path to a .mlir attribute file">,
    ];
    let dependentDialects = [
        "::mlir::seki::SekiDialect",
        "::mlir::seki_hw::SekiHwDialect",
    ];
}

def MemorySpaceAssignmentPass : Pass<"seki-assign-memory-spaces", "::mlir::ModuleOp"> {
    let summary = "Promote scatchpad-sized memref.alloc ops to scratchpad memory space";
    let description = [{
        Walks all memref.alloc ops in the module and promotes those that fit within the target
        scratchpad budget to the scratchpad memory space, setting the DMA alignment attribute accodingly.
        Requires seki-attach-target to have run first.
    }];
    let dependentDialects = [
        "::mlir::seki::SekiDialect",
        "::mlir::memref::MemRefDialect",
    ];

}

def SekiLowerToSekiHwPass : Pass<"seki-lower-to-seki-hw", "::mlir::func::FuncOp"> {
    let summary = "Lower seki dialect ops to seki_hw dialect ops";
    let description = [{
        Converts seki dialect ops to their seki_hw equivalents using ConversionTarget + TypeConverter
        to handle the tensor<MxNxT> -> !seki_hw.tile<T> type boundary.
    }];
    let dependentDialects = [
        "::mlir::seki::SekiDialect",
        "::mlir::seki_hw::SekiHwDialect",
        "::mlir::func::FuncDialect"
    ];
}

def SekiVerifyStaticShapesPass : Pass<"seki-verify-static-shapes", "::mlir::func::FuncOp"> {
    let summary = "Error on any dynamic tensor dimension; enforces static-shapes-only contract";
    let description = [{
        Walks all ops in the function. Emits an error for every tensor-typed operand or result whose shape
        contains a dynamic dimension (?). Must run immediately after stablehlo-to-linalg and before --seki-lift-to-seki.
    }];
    let dependentDialects = ["::mlir::func::FuncDialect"];
}

def SekiLiftToSekiPass : Pass<"seki-lift-to-seki", "::mlir::func::FuncOp">{
    let summary = "Lift linalg.* ops to seki.* named ops";
    let description = [{
        Recognises linalg compute and shape ops by semantic structure and replaces them with named seki.*
        equivalents. Unrecognised linalg generics produce a compile-time error with the op location.
    }];
    let dependentDialects = ["::mlir::seki::SekiDialect"];
}

def SekiCastToTargetTypesPass : Pass<"seki-cast-to-target-types", "::mlir::ModuleOp"> {
    let summary = "Insert seki.cast ops to narrow matmul inputs to the MAC array's accepted type";
    let description = [{
        Reads macInputType() from SekiTargetInfo. Walks seki.matmul and seki.batch_matmul;
        for each ins operand whose element type does not match the MAC input type, inserts a seki.cast immediately
        before the op. Guards each insertion with isCastSupported(from, to) and signals pass failure if no cast path
        exists. The outs operand is not modified, its type drives the hardware output mode.
        This pass is opt-in and off by default; run before --seki-check-hw-units.
    }];
    let dependentDialects = [
        "::mlir::seki::SekiDialect",
        "::mlir::seki_hw::SekiHwDialect",
    ];
}

def SekiCheckHwUnitsPass : Pass<"seki-check-hw-units", "::mlir::ModuleOp"> {
    let summary = "Verify that all seki ops are supported by the target's hardware units";
    let description = [{
        Walks every op in the module and checks 2 things per op:
        1. the required hardware unit exists on the target, and
        2. the op's types are legal for that unit.
        Pure verifier - no IR is modified, signals failure on the first unsatisfied constraint.
        seki.cast is handled separately via isCastSupported().
    }];
    let dependentDialects = ["mlir::seki::SekiDialect", "mlir::seki_hw::SekiHwDialect"];
}

#endif // SEKI_PASSES_TD
