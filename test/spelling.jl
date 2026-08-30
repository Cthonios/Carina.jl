# US spelling guard.
#
# Carina standardized on US spelling in 8ccc8e9 (45 substitutions across 31
# files).  Prose drifts back one commit at a time, and a spelling nobody
# notices is exactly the kind of thing that reappears, so this scans the
# tracked sources on every run.
#
# The point is not orthography for its own sake.  One of the spellings that
# commit fixed had the docs writing `travelling` while the deck key the parser
# accepts is `traveling wave` -- documentation disagreeing with the input a
# user actually types.

@testset "US spelling" begin

    # British -> American.  Inflections are enumerated rather than derived from
    # stems: a stem-based pattern would fire on `realistic`, `characteristic`,
    # `cancellation` and `analysis`, all of which are correct US spellings that
    # occur in this repository.  See the self-check below.
    BRITISH = Dict(
        "behaviour" => "behavior", "behaviours" => "behaviors",
        "neighbour" => "neighbor", "neighbours" => "neighbors",
        "neighbouring" => "neighboring", "neighbourhood" => "neighborhood",
        "honour" => "honor", "honours" => "honors", "honoured" => "honored",
        "honouring" => "honoring",
        "favour" => "favor", "favours" => "favors", "favoured" => "favored",
        "favourable" => "favorable", "favourably" => "favorably",
        "colour" => "color", "colours" => "colors", "coloured" => "colored",
        "recognise" => "recognize", "recognises" => "recognizes",
        "recognised" => "recognized", "recognising" => "recognizing",
        "unrecognised" => "unrecognized", "unrecognisable" => "unrecognizable",
        "normalise" => "normalize", "normalises" => "normalizes",
        "normalised" => "normalized", "normalising" => "normalizing",
        "normalisation" => "normalization",
        "initialise" => "initialize", "initialises" => "initializes",
        "initialised" => "initialized", "initialising" => "initializing",
        "initialisation" => "initialization",
        "capitalise" => "capitalize", "capitalised" => "capitalized",
        "capitalisation" => "capitalization",
        "realise" => "realize", "realises" => "realizes",
        "realised" => "realized", "realising" => "realizing",
        "optimise" => "optimize", "optimised" => "optimized",
        "optimising" => "optimizing", "optimisation" => "optimization",
        "discretise" => "discretize", "discretised" => "discretized",
        "discretisation" => "discretization",
        "minimise" => "minimize", "minimised" => "minimized",
        "maximise" => "maximize", "maximised" => "maximized",
        "characterise" => "characterize", "characterised" => "characterized",
        "generalise" => "generalize", "generalised" => "generalized",
        "generalisation" => "generalization",
        "specialise" => "specialize", "specialised" => "specialized",
        "penalise" => "penalize", "penalised" => "penalized",
        "summarise" => "summarize", "summarised" => "summarized",
        "emphasise" => "emphasize", "emphasised" => "emphasized",
        "organise" => "organize", "organised" => "organized",
        "serialise" => "serialize", "serialised" => "serialized",
        "synchronise" => "synchronize", "synchronised" => "synchronized",
        "analyse" => "analyze", "analysed" => "analyzed",
        "analysing" => "analyzing",
        "analogue" => "analog", "analogues" => "analogs",
        "centre" => "center", "centres" => "centers", "centred" => "centered",
        "travelling" => "traveling", "travelled" => "traveled",
        "traveller" => "traveler",
        "cancelling" => "canceling", "cancelled" => "canceled",
        "modelling" => "modeling", "modelled" => "modeled",
        "labelling" => "labeling", "labelled" => "labeled",
        "whilst" => "while", "amongst" => "among",
        "artefact" => "artifact", "artefacts" => "artifacts",
        "judgement" => "judgment", "acknowledgement" => "acknowledgment",
        "fulfil" => "fulfill", "fulfilment" => "fulfillment",
        "grey" => "gray", "programme" => "program", "defence" => "defense",
        "licence" => "license", "practise" => "practice",
        "catalogue" => "catalog", "dialogue" => "dialog",
        "orientated" => "oriented", "learnt" => "learned", "spelt" => "spelled",
        "litre" => "liter", "metre" => "meter", "fibre" => "fiber",
        "theatre" => "theater", "sombre" => "somber", "spectre" => "specter",
        "mould" => "mold", "draught" => "draft", "plough" => "plow",
        "aluminium" => "aluminum", "sulphur" => "sulfur",
        "storey" => "story", "ageing" => "aging", "skilful" => "skillful",
        "focussed" => "focused", "sceptical" => "skeptical",
        "manoeuvre" => "maneuver", "connexion" => "connection",
    )

    PATTERN = Regex("\\b(" *
        join(sort(collect(keys(BRITISH)), by = length, rev = true), "|") *
        ")\\b", "i")

    # Prose only.  `.bib` is deliberately absent: a bibliography holds published
    # titles, journal names and author names verbatim, and "correcting" one
    # would misquote the source.
    SCANNED_EXTENSIONS = (".jl", ".md", ".tex", ".yaml", ".yml",
                                ".toml", ".sh")

    # This file necessarily contains every word it forbids, so it must exclude
    # itself or the guard fails the moment it is added.
    EXCLUDED = ("test/spelling.jl",)

    @testset "the pattern does not fire on correct US spellings" begin
        # Every one of these occurs in the repository and shares a prefix with
        # a forbidden word.  If a future edit to BRITISH makes the pattern
        # stem-based, this fails before it can produce a wave of false reports.
        for good in ("realistic", "characteristic", "cancellation", "analysis",
                     "analyses", "exercise", "precise", "concise", "promise",
                     "otherwise", "likewise", "noise", "raise", "expertise",
                     "premise", "revise", "devise", "compromise", "surprise",
                     "modelled_by_nothing", "central", "concentrate",
                     "hour", "four", "your", "flour", "colourless_is_not_a_word")
            @test !occursin(Regex("^(" * PATTERN.pattern * ")\$", "i"), good)
        end
        # ...and it does fire on the words it is for.
        for bad in ("behaviour", "Behaviour", "NEIGHBOUR", "unrecognised",
                    "travelling", "analogue")
            @test occursin(PATTERN, bad)
        end
    end

    @testset "tracked sources use US spelling" begin
        root = normpath(joinpath(@__DIR__, ".."))
        files = try
            out = read(Cmd(`git ls-files`, dir = root), String)
            filter(!isempty, split(out, '\n'))
        catch
            String[]   # not a git checkout (tarball, vendored copy)
        end

        if isempty(files)
            # Degrade to a skip rather than a failure: a source tree without
            # git is a legitimate way to run this suite, and a guard that fails
            # there teaches people to ignore it.
            @test_skip "git ls-files unavailable; spelling scan skipped"
        else
            offenses = String[]
            for f in files
                any(e -> endswith(f, e), SCANNED_EXTENSIONS) || continue
                f in EXCLUDED && continue
                path = joinpath(root, f)
                isfile(path) || continue
                for (n, line) in enumerate(eachline(path))
                    for m in eachmatch(PATTERN, line)
                        w = m.match
                        push!(offenses,
                              "$f:$n: \"$w\" -> \"$(BRITISH[lowercase(w)])\"")
                    end
                end
            end
            if !isempty(offenses)
                Carina._carina_log(0, :warning, "British spellings found:")
                for o in offenses
                    Carina._carina_log(0, :warning, "  " * o)
                end
            end
            @test isempty(offenses)
        end
    end
end
