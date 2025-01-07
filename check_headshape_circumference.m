%__________________________________________________________________________
% Script to check the circumference of a 3D model head scan from the
% Einscan

% Authors:  Robert Seymour      (rob.seymour@psych.ox.ac.uk)    
%__________________________________________________________________________

cd('/Users/robertseymour/data/20241216_oxford_pilot');
addpath(which("estimate_circumference.m"))

headshape = ft_read_headshape('seamus_head.ply');
headshape = ft_convert_units(headshape,'mm');

% Realign the headshape to the Neuromag coordinate system
cfg                 = [];
cfg.method          = 'fiducial';
cfg.coordsys        = 'neuromag';
headshape_realign   = ft_meshrealign(cfg, headshape);

circumference = estimate_circumference(headshape_realign, 30)


















