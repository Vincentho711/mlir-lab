module {
    // Integer types of different widths
    func.func @integer_types() -> i64 {
        %a = arith.constant 100 : i8
        %a_2 = arith.constant 100 : i8
        %b = arith.constant 200 : i32
        %c = arith.constant 300 : i64

        %a_to_i32 = arith.extsi %a : i8 to i32
        %sum_i32 = arith.addi %a_to_i32, %b : i32

        %sum_to_i64 = arith.extsi %sum_i32 : i32 to i64
        %result = arith.addi %sum_to_i64, %c : i64

        func.return %result : i64
    }

    // Float types and float operations
    func.func @float_types() -> f64 {
        %x = arith.constant 1.5 : f32
        %y = arith.constant 2.5 : f64

        %x_to_f64 = arith.extf %x : f32 to f64
        %result   = arith.addf %x_to_f64, %y : f64

        func.return %result : f64
    }

    // Converting between integers and floats
    func.func @int_to_float() -> f32 {
        %n = arith.constant 52 : i32
        %result = arith.sitofp %n : i32 to f32

        func.return %result : f32
    }

    // Index type
    func.func @index_type() -> index {
        %i = arith.constant 10 : index
        %j = arith.constant 20 : index
        %result = arith.addi %i, %j : index

        func.return %result : index
    }

    // Attributes vs SSA values
    func.func @attributes_vs_values() -> i32 {
        %c = arith.constant 42 : i32
        %t = arith.constant true
        %f = arith.constant false

        func.return %c : i32
    }

}
