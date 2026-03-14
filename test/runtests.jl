using Statistics
using Test

include("../src/Carina.jl")
include("helpers.jl")

# ---------------------------------------------------------------------------
# Test registry — add new tests here in order.
# ---------------------------------------------------------------------------

const indexed_test_files = [
    (1, "mechanics-quasistatic-cube.jl"),
    (2, "mechanics-implicit-dynamic-cube.jl"),
    (3, "mechanics-explicit-dynamic-cube.jl"),
]

# ---------------------------------------------------------------------------
# Argument parsing  (same flags as Norma: --list, --filter, --quick, indices)
# ---------------------------------------------------------------------------

function print_available_tests()
    Carina._carina_log(0, :setup, "Available tests:")
    for (i, file) in indexed_test_files
        Carina._carina_log(0, :setup, rpad("[$i]", 5) * file)
    end
end

function parse_args(args)
    if "--list" in args
        print_available_tests()
        exit(0)
    end

    filter_idx  = findfirst(isequal("--filter"), args)
    name_filter = filter_idx !== nothing && filter_idx < length(args) ?
                  lowercase(args[filter_idx + 1]) : ""

    quick_only = "--quick" in args

    selected_indices = try
        parse.(Int, filter(x -> occursin(r"^\d+$", x), args))
    catch
        Carina._carina_log(0, :warn, "Invalid test index.")
        exit(1)
    end

    candidate_tests = if !isempty(selected_indices)
        valid = Set(i for (i, _) in indexed_test_files)
        for i in selected_indices
            if i ∉ valid
                Carina._carina_log(0, :warn, "Invalid test index: $i")
                exit(1)
            end
        end
        filter(t -> t[1] in selected_indices, indexed_test_files)
    else
        indexed_test_files
    end

    if !isempty(name_filter)
        candidate_tests = filter(t -> occursin(name_filter, lowercase(t[2])), candidate_tests)
        if isempty(candidate_tests)
            Carina._carina_log(0, :warn, "No tests match filter \"$name_filter\".")
            exit(1)
        end
    end

    return candidate_tests
end

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

tests_to_run = parse_args(ARGS)

start_time = time()
Carina._carina_log(0, :carina, "BEGIN TESTS")

@testset verbose=true "Carina.jl Test Suite" begin
    for (i, file) in tests_to_run
        Carina._carina_log(0, :setup, "[$i] $file")
        t0 = time()
        include(file)
        Carina._carina_logf(0, :time, "[$i] wall = %.1fs", time() - t0)
    end
end

Carina._carina_logf(0, :time, "Total wall time = %.1fs", time() - start_time)
Carina._carina_log(0, :carina, "END TESTS")
