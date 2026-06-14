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
    let dependentDialects = ["::mlir::seki::SekiDialect"];
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
#endif // SEKI_PASSES_TD
