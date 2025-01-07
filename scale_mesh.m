%__________________________________________________________________________
% Script to check the circumference of a 3D model head scan from the
% Einscan

% Authors:  Robert Seymour      (rob.seymour@psych.ox.ac.uk)    
%__________________________________________________________________________

addpath('C:\Users\Cerca Stim\Documents\MATLAB\fieldtrip-20240110');
ft_defaults
[pathstr, name, ext] = fileparts(which("estimate_circumference.m"));
addpath(pathstr)

%% File
cd('C:\Users\Cerca Stim\Documents\EXScan H\20241216\seamus_helmet_head');

file = 'seamus_helmet_head.ply';
[~, name, ext] = fileparts(file);

headshape = ft_read_headshape(file);
headshape = ft_convert_units(headshape,'mm');

%% Scale mesh to 97.5%
% tape_measure_value = 565; % in mm
% scaling            = tape_measure_value/circumference;
scaling            = 0.975;
disp(['Scaling to: ' num2str(scaling*100) '%'])

trans_matrix = [scaling 0 0 0;...
    0 scaling 0 0; 0 0 scaling 0; 0 0 0 1];

headshape_scaled = ft_transform_geometry(trans_matrix,headshape);

figure; ft_plot_headshape(headshape,'facealpha',0.1); hold on;
ft_plot_headshape(headshape_scaled,'facealpha',1); hold on;
title('Scaled Headshape');


%% Export 
headshape_to_ply(headshape_scaled,[name '_SCALED.ply'])