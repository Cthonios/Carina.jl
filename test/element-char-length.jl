@testset "Element Characteristic Length" begin
    # element_char_length must return the minimum node-pair distance, which is
    # the shortest edge for well-shaped elements.  The previous implementation
    # returned ~2× the mean centroid-to-node distance — the body diagonal √3·h
    # for a cube — overestimating the explicit stable time step by ~1.73×.

    physics = Carina.SolidMechanics(Carina.CM.NeoHookean(), 1000.0)
    h = 2.5e-3

    call(x_el, u_el) = Carina.element_char_length(
        physics, nothing, x_el,
        0.0, 0.0, u_el, u_el,
        nothing, nothing, nothing,
    )

    # HEX8 cube with edge h (Exodus node ordering, interleaved DOFs).
    hex = h .* [
        0.0 0.0 0.0
        1.0 0.0 0.0
        1.0 1.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0
        1.0 0.0 1.0
        1.0 1.0 1.0
        0.0 1.0 1.0
    ]
    x_hex = vec(hex')
    @test call(x_hex, zero(x_hex)) ≈ h rtol = 1e-12

    # Uses current (deformed) coordinates: uniform stretch u = α·X.
    α = 0.1
    @test call(x_hex, α .* x_hex) ≈ (1 + α) * h rtol = 1e-12

    # TET4 right-corner tet: axis edges h, diagonals √2·h → min edge h.
    tet = h .* [
        0.0 0.0 0.0
        1.0 0.0 0.0
        0.0 1.0 0.0
        0.0 0.0 1.0
    ]
    x_tet = vec(tet')
    @test call(x_tet, zero(x_tet)) ≈ h rtol = 1e-12
end
