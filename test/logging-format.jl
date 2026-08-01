# Console/file logging plumbing: the pieces a normal simulation run never
# touches.  CI exports CI=true, which switches _use_color() off globally, so
# the ANSI path, the fallback lookups, the log-file mirror, and the
# CARINA_TIMING instrumentation were all invisible to the suite -- a broken
# escape code or a log file that never flushed would have reached users first.

@testset "Logging and formatting" begin

    # ----- lookup tables and their fallbacks --------------------------------
    @testset "indent / label / ansi lookups" begin
        @test Carina._indent(0)  == ""
        @test Carina._indent(4)  == "    "
        @test Carina._indent(8)  == "        "
        # Levels that are not a multiple of 4 (or beyond the table) fall back
        # to flush-left rather than erroring mid-log.
        @test Carina._indent(3)  == ""
        @test Carina._indent(99) == ""

        @test Carina._label(:setup)   == "[SETUP]  "
        @test Carina._label(:warning) == "[WARNING]"
        # Unknown keywords synthesize a bracketed label instead of erroring.
        @test Carina._label(:frobnicate) == "[FROBNICATE]"

        @test Carina._ansi(:green) == "\e[32m"
        # Unknown colors fall back to the terminal default foreground.
        @test Carina._ansi(:mauve) == "\e[39m"
    end

    # ----- wall-time formatting ---------------------------------------------
    @testset "format_time" begin
        @test Carina.format_time(12.34)   == "12.34s"
        @test Carina.format_time(154.56)  == "2m 34.56s"
        @test Carina.format_time(5025.67) == "1h 23m 45.67s"
        @test Carina.format_time(93784.56) == "1d 2h 3m 4.56s"
        # Zero-valued middle units are dropped, not printed as "0m".
        @test Carina.format_time(3601.0)  == "1h 1.00s"
    end

    # ----- color gate and status strings ------------------------------------
    @testset "status strings honor the color gate" begin
        colorless = ("CI" => "true",)
        colorful  = ("CI" => nothing, "NO_COLOR" => nothing,
                     "CARINA_NO_COLOR" => nothing, "FORCE_COLOR" => "1")

        withenv(colorless...) do
            @test !Carina._use_color()
            @test Carina._status_str(true)     == "[DONE]"
            @test Carina._status_str(false)    == "[WAIT]"
            @test Carina._cg_status_str(true)  == "[CONV]"
            @test Carina._cg_status_str(false) == "[STALL]"
        end

        withenv(colorful...) do
            @test Carina._use_color()
            @test occursin("\e[32m", Carina._status_str(true))
            @test occursin("\e[33m", Carina._status_str(false))
            @test occursin("\e[32m", Carina._cg_status_str(true))
            @test occursin("\e[31m", Carina._cg_status_str(false))
            # The colored console branch of _carina_log (bold + color + reset).
            Carina._carina_log(0, :setup, "color path smoke test")
        end
    end

    # ----- log-file mirror ---------------------------------------------------
    # runtests.jl switches CARINA_WRITE_LOG_FILE off for the whole suite so
    # test runs do not litter .log files; opt back in for this block only.
    @testset "log file open/write/close" begin
        mktempdir() do dir
            input = joinpath(dir, "run.yaml")

            Carina.CARINA_WRITE_LOG_FILE[] = true
            try
                Carina.open_log_file(input)
                @test Carina.CARINA_LOG_FILE[] !== nothing
                # A second open is a no-op: the outermost run() owns the file.
                Carina.open_log_file(joinpath(dir, "other.yaml"))
                @test !isfile(joinpath(dir, "other.log"))

                Carina._carina_log(4, :solve, "mirrored line")
                Carina.close_log_file()
                @test Carina.CARINA_LOG_FILE[] === nothing
                # Closing twice must be harmless.
                Carina.close_log_file()
            finally
                Carina.CARINA_WRITE_LOG_FILE[] = false
                Carina.close_log_file()
            end

            log = read(joinpath(dir, "run.log"), String)
            @test occursin("[SOLVE]", log)
            @test occursin("mirrored line", log)
            # The file mirror is stripped of ANSI escapes.
            @test !occursin("\e[", log)
        end

        # With the global switch off, no file is created at all.
        mktempdir() do dir
            Carina.open_log_file(joinpath(dir, "run.yaml"))
            @test Carina.CARINA_LOG_FILE[] === nothing
            @test !isfile(joinpath(dir, "run.log"))
        end
    end

    # ----- phase / timing macros --------------------------------------------
    @testset "carina_phase and carina_timed" begin
        # Without CARINA_TIMING both macros are announce-only pass-throughs.
        withenv("CARINA_TIMING" => nothing) do
            @test !Carina._carina_timing_on()
            @test Carina.@carina_phase("untimed phase", 1 + 1) == 2
            @test Carina.@carina_timed("untimed block", 2 + 2) == 4
        end

        # With CARINA_TIMING set they time the block and still return its value.
        withenv("CARINA_TIMING" => "1") do
            @test Carina._carina_timing_on()
            @test Carina.@carina_phase("timed phase", 3 + 3) == 6
            @test Carina.@carina_timed("timed block", 4 + 4) == 8
        end
    end
end
