module {
    // Explicit control flow with the cf dialet.
    // Multiple blocks, block arguments, and branches.
    func.func @cf_min(%a: i32, %b: i32) -> i32 {
        %cond = arith.cmpi slt, %a, %b : i32
        cf.cond_br %cond, ^pick_a, ^pick_b

        ^pick_a:
            cf.br ^exit(%a : i32)

        ^pick_b:
            cf.br ^exit(%b : i32)

        ^exit(%result : i32):
            func.return %result : i32
    }

    // Structured conditional using scf.if (high level), scf (structured control flow)
    func.func @scf_max(%a : i32, %b : i32) -> i32 {
        %cond = arith.cmpi sgt, %a, %b : i32
        %result = scf.if %cond -> (i32) {
            scf.yield %a : i32
        } else {
            scf.yield %b : i32
        }
        func.return %result : i32
    }

    // Structured loop using scf.for
    // Computes the sum 0 + 1 + 2 + ... + (n-1)
    func.func @sum_to_n(%n: index) -> i32 {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %zero = arith.constant 0 : i32

        %result = scf.for %i = %c0 to %n step %c1
                  iter_args(%sum = %zero) -> (i32) {
            %i_i32 = arith.index_cast %i : index to i32
            %new_sum = arith.addi %sum, %i_i32 : i32
            scf.yield %new_sum : i32
        }
        func.return %result : i32
    }
}
