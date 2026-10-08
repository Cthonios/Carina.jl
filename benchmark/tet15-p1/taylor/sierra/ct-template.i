# Taylor bar impact (Foulk et al. 2021, Sec. 4.4) with the composite
# tetrahedron of Sierra/SM, explicit dynamics.  Units m, kg, s.
#
# Copper bar, axis z, length 32.4 mm, radius 3.2 mm; rigid frictionless wall
# at z = 0 as u_z = 0 on node set "impact"; initial velocity -227 m/s in z on
# node set "all".  Density 8930, E = 117 GPa, nu = 0.35; J2 plasticity with
# linear isotropic hardening (LAME fefp, as in the deck of the paper), yield
# stress 400 MPa, hardening modulus 100 MPa.  No bulk viscosity.
#
# Placeholders, substituted by run-ct.sh:
#   {MESH}    TETRA10 mesh (meshes/taylor-h<h>-tet10.g)
#   {OUT}     results file
#   {HB}      heartbeat file
#   VEM       (in braces, on a line of its own) further section lines of the VEM stabilization (empty: alpha
#             and the bulk parameter at Sierra's defaults, 0.1 and 1.0e-6)
#
# vem exponent = 0.0 as in the deck of the paper (Sierra's default is 5.0).

begin sierra taylor_bar

  begin material copper
    density = 8930.0
    begin parameters for model fefp
      youngs modulus = 117.0e9
      poissons ratio = 0.35
      yield stress = 0.4e9
      hardening model = linear
      hardening modulus = 0.1e9
      max_ls_iter = 1000
      max_rma_iter = 1000
    end
  end material copper

  begin total lagrange section ct
    formulation = composite_tet
    vem exponent = 0.0
{VEM}
  end

  begin finite element model taylor
    Database Name = {MESH}
    Database Type = exodusII
    decomposition method = rcb

    begin parameters for block bar
      material = copper
      model = fefp
      section = ct
      linear bulk viscosity = 0.0
      quadratic bulk viscosity = 0.0
    end parameters for block bar
  end finite element model taylor

  begin presto procedure taylor_procedure

    begin time control
      begin time stepping block p1
        start time = 0.0
        begin parameters for presto region taylor_region
          step interval = 1000
        end parameters for presto region taylor_region
      end time stepping block p1
      termination time = 80.0e-6
    end time control

    begin presto region taylor_region
      use finite element model taylor

      begin fixed displacement
        node set = impact
        component = z
      end

      begin initial velocity
        node set = all
        component = z
        magnitude = -227.0
      end

      # fefp requires a temperature field; none of its constants here depend
      # on it (the symmetric deck of the paper sets the same value).
      begin initial temperature
        block = bar
        magnitude = 273.15
      end

      begin user output
        node set = impact
        compute global wall_force as sum of nodal reaction(z)
      end

      begin user output
        include all blocks
        compute global max_eqps as max of element eqps
      end

      begin Heartbeat Output taylor_heartbeat
        stream name = {HB}
        At Time 0.0, Increment = 1.0e-6
        precision = 8
        labels = off
        legend = on
        timestamp format = ""
        global time
        global timestep
        global max_eqps
        global wall_force
        global kinetic_energy
        global internal_energy
        global external_energy
      end

      begin solution termination
        terminate global timestep < 1.0e-12
      end

      begin Results Output taylor_output
        Database Name = {OUT}
        Database Type = exodusII
        At Time 0.0, Increment = 1.0e-6
        nodal variables = displacement
        nodal variables = velocity
        element variables = eqps
        element variables = von_mises
        global variables = timestep
        global variables = kinetic_energy
        global variables = internal_energy
      end

    end presto region taylor_region
  end presto procedure taylor_procedure
end sierra taylor_bar
