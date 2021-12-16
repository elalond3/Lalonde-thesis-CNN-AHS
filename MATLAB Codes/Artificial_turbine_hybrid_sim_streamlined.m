function Artificial_turbine_hybrid_sim_streamlined

%% Description
% This program will test the rotational residential wind turbine model dervied from aeroelastic wind
% tunnel testing equipped with a semi-active variable-damping TMD to three sets of wind loads using
% four different aerodynamic models: 1. FAST TH of a fixed turbine; 2. FAST TH of a rotational
% turbine; 3. FAST TH of a rotational turbine with PTMD; 4. CFD-based neural network. The results
% from these four sims will be compared

% The three wind speed cases are from my wind tunnel testing (3.63 m/s mean, 6.7% turbulence, 0.95
% TSR, 3.14 rad/s rotor speed); Kamran's testing (5.15 m/s mean, 10% turbulence, 1.08 TSR, 5.06 
% rad/s rotor speed); and the recommended operational conditions of the wind turbine (9.00 m/s mean,
% 10% turbulence, 5.20 TSR, 42.54 rad/s rotor speed)

% Turbine model is approximated as a 1 DOF inverted pendulum. The weight of the turbine is simulated
% as a lumped mass at the end of the tower and the hub height wind speed loads both the lumped mass
% rotor and the tower. The lumped mass is also shifted forward by a value of d, this eccentricity is
% due to the fact that the COM of the turbine isn't over the tower center.
% mr, Er, kr are the rotational mass, damping ratio, and stiffness of the system
% theta is base rotation
% h is tower height and d is effective mass overhang

%      m d
%      O---|
%          |
%          |  h
%       <- | ->
%    (k,E) | theta
% ---------O----------


