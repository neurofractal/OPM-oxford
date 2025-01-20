addpath('C:\Users\Cerca Stim\Documents\MATLAB\parallel_port\');
addpath(genpath('C:\Users\Cerca Stim\Documents\NatNetSDK'))

clear all; clc
%% I/O
io_obj = io64;
status = io64(io_obj);

%% Values to Change
address = 16376;  % Address of the trigger
trigger_val_to_send = 32;  % Trigger value to send

%% Generate filename
task = 'techDev';
participant = '000';
session = '000';
run = '000';

% Get the current date and time in the required format
currentTime = datetime('now', 'Format', 'yyyyMMdd_HH_mm_ss');

% Generate the filename
filename = sprintf('sub-%s_task-%s_session-%s_run-%s_%s_motion.tak', ...
    participant, task, session, run, currentTime);

%%
% Reset to 0 (prepare for trigger)
io64(io_obj, address, 0);

% Create NatNet Client
myClientObject = natnet();
myClientObject.connect;

% Start recording when the user types something to start
disp('Type anything in the command window to start recording...');
input('Press Enter to start recording...');  % Wait for user to press Enter

% Print the address and trigger value just before starting
disp(['Starting recording...']);
disp(['Address: ', num2str(address)]);
disp(['Trigger Value: ', num2str(trigger_val_to_send)]);

% Send trigger and start recording
io64(io_obj, address, trigger_val_to_send);  % Send the trigger signal

% Set take name
myClientObject.setTakeName(filename);

% Start recording
myClientObject.startRecord;

disp('Recording started. Type anything again to stop.');

% Wait for user to type something to stop
input('Press Enter to stop recording...');  % Wait for user to press Enter again

% Stop recording and reset trigger
disp('Stopping recording...');
myClientObject.stopRecord;
io64(io_obj, address, 0);  % Reset the trigger signal

disp('STOP');
% Disconnect from NatNet at the end
myClientObject.disconnect;

