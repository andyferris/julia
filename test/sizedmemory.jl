# Tests for SizedMemory{T, n}

@testset "SizedMemory" begin

    @testset "construction" begin
        m = SizedMemory{Float64, 3}()
        @test length(m) == 3
        @test size(m) == (3,)
        @test eltype(m) === Float64

        m0 = SizedMemory{Int, 0}()
        @test length(m0) == 0
        @test size(m0) == (0,)

        # Zero-length singleton (following Memory pattern)
        @test SizedMemory{Int, 0}() === SizedMemory{Int, 0}()
        @test SizedMemory{Float64, 0}() === SizedMemory{Float64, 0}()
        # Different T → different singleton
        @test !(SizedMemory{Int, 0}() === SizedMemory{Float64, 0}())

        # Non-zero-length has identity
        @test !(SizedMemory{Int, 1}() === SizedMemory{Int, 1}())

        # Undef constructor
        m = SizedMemory{Int, 5}(undef)
        @test length(m) == 5
    end

    @testset "isbits element types" begin
        # Int
        m = SizedMemory{Int, 3}()
        m[1] = 10; m[2] = 20; m[3] = 30
        @test m[1] == 10
        @test m[2] == 20
        @test m[3] == 30

        # Float64
        m = SizedMemory{Float64, 2}()
        m[1] = 1.5; m[2] = 2.5
        @test m[1] === 1.5
        @test m[2] === 2.5

        # Bool
        m = SizedMemory{Bool, 4}()
        m[1] = true; m[2] = false; m[3] = true; m[4] = false
        @test m[1] === true
        @test m[3] === true

        # ComplexF64
        m = SizedMemory{ComplexF64, 2}()
        m[1] = 1.0 + 2.0im
        @test m[1] === 1.0 + 2.0im

        # Int8
        m = SizedMemory{Int8, 5}()
        m[1] = Int8(42)
        @test m[1] === Int8(42)
    end

    @testset "boxed element types (Any, mutable struct, abstract)" begin
        # Any
        m = SizedMemory{Any, 3}()
        m[1] = 42
        m[2] = "hello"
        m[3] = [1, 2, 3]
        @test m[1] == 42
        @test m[2] == "hello"
        @test m[3] == [1, 2, 3]

        # mutable struct
        mutable struct SMTestMut
            x::Int
        end
        m = SizedMemory{SMTestMut, 2}()
        a = SMTestMut(1)
        b = SMTestMut(2)
        m[1] = a; m[2] = b
        @test m[1] === a
        @test m[2] === b
        a.x = 99
        @test m[1].x == 99

        # Abstract type
        m = SizedMemory{Number, 2}()
        m[1] = 42
        m[2] = 3.14
        @test m[1] == 42
        @test m[2] == 3.14

        # Unassigned element throws UndefRefError
        m = SizedMemory{Any, 2}()
        @test_throws UndefRefError m[1]
        m[1] = nothing
        @test m[1] === nothing
    end

    @testset "isbits union element types" begin
        m = SizedMemory{Union{Int,Float64}, 3}()
        m[1] = 42
        m[2] = 3.14
        m[3] = 100
        @test m[1] === 42
        @test m[2] === 3.14
        @test m[3] === 100

        m = SizedMemory{Union{Int,Nothing}, 2}()
        m[1] = 42
        m[2] = nothing
        @test m[1] === 42
        @test m[2] === nothing
    end

    @testset "immutable struct with pointers" begin
        m = SizedMemory{Pair{Int, String}, 2}()
        m[1] = 1 => "one"
        m[2] = 2 => "two"
        @test m[1] == (1 => "one")
        @test m[2] == (2 => "two")
    end

    @testset "zero-sized element types" begin
        m = SizedMemory{Nothing, 3}()
        @test length(m) == 3
        @test m[1] === nothing
        @test m[2] === nothing

        # Zero-sized T with n > 0 → singleton (no mutable state)
        @test SizedMemory{Nothing, 3}() === SizedMemory{Nothing, 3}()

        # Zero-length AND zero-sized → singleton
        @test SizedMemory{Nothing, 0}() === SizedMemory{Nothing, 0}()
    end

    @testset "bounds checking" begin
        m = SizedMemory{Int, 3}()
        @test_throws BoundsError m[0]
        @test_throws BoundsError m[4]
        @test_throws BoundsError m[-1]

        m0 = SizedMemory{Int, 0}()
        @test_throws BoundsError m0[1]

        # @inbounds skips check
        m[1] = 42
        @test @inbounds(m[1]) == 42
    end

    @testset "invalid type parameters" begin
        # Wrong parameter order: n first, T second
        @test_throws TypeError SizedMemory{3, Float64}()
        # Negative n
        @test_throws TypeError SizedMemory{Float64, -1}()
        # Non-type T with integer n
        @test_throws TypeError SizedMemory{42, 3}()
    end

    @testset "type checking on setindex!" begin
        m = SizedMemory{Float64, 2}()
        # Conversion should work
        m[1] = 42  # Int → Float64
        @test m[1] === 42.0
    end

    @testset "sizeof and layout" begin
        @test sizeof(SizedMemory{Float64, 3}()) == 24
        @test sizeof(SizedMemory{Int32, 4}()) == 16
        @test sizeof(SizedMemory{Int, 0}()) == 0
        @test sizeof(SizedMemory{Nothing, 100}()) == 0
    end

    @testset "iteration" begin
        m = SizedMemory{Int, 3}()
        m[1] = 10; m[2] = 20; m[3] = 30
        @test collect(m) == [10, 20, 30]

        m0 = SizedMemory{Int, 0}()
        @test collect(m0) == Int[]
    end

    @testset "copy" begin
        m = SizedMemory{Int, 3}()
        m[1] = 1; m[2] = 2; m[3] = 3
        m2 = copy(m)
        @test m2[1] == 1 && m2[2] == 2 && m2[3] == 3
        @test !(m === m2)
        m2[1] = 99
        @test m[1] == 1
    end

    @testset "type parameter inference" begin
        f() = length(SizedMemory{Float64, 7}())
        @test @inferred(f()) == 7
    end

    @testset "GC stress" begin
        # Boxed elements survive GC
        m = SizedMemory{Any, 100}()
        for i in 1:100
            m[i] = "element_$i"
        end
        GC.gc()
        for i in 1:100
            @test m[i] == "element_$i"
        end

        # Mutable struct elements survive GC
        mutable struct SMGCTest
            v::Vector{Int}
        end
        m = SizedMemory{SMGCTest, 10}()
        for i in 1:10
            m[i] = SMGCTest(fill(i, 100))
        end
        GC.gc(); GC.gc()
        for i in 1:10
            @test m[i].v == fill(i, 100)
        end
    end

    @testset "large n" begin
        m = SizedMemory{Float64, 1000}()
        for i in 1:1000
            m[i] = Float64(i)
        end
        @test m[500] === 500.0
        @test m[1000] === 1000.0
    end

    @testset "nested SizedMemory" begin
        m = SizedMemory{SizedMemory{Int, 2}, 3}()
        for i in 1:3
            inner = SizedMemory{Int, 2}()
            inner[1] = i * 10
            inner[2] = i * 10 + 1
            m[i] = inner
        end
        @test m[2][1] == 20
        @test m[3][2] == 31
    end

end  # @testset "SizedMemory"
