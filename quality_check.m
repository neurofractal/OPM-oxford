%__________________________________________________________________________
% Quality checking script for the Oxford OPM Lab
% 
% Authors:  Robert Seymour      (rob.seymour@psych.ox.ac.uk)
%
% Dec 2024
%__________________________________________________________________________

%% Get the directories in order
script_dir      = '/Users/robertseymour/scripts/analyse_OPMEG';
spm_path        = '/Users/robertseymour/scripts/spm-main'

% Add analyse_OPMEG to path
addpath(genpath(script_dir));

% BIDS data directory:
data_dir        = '/Users/robertseymour/data/20241216_oxford_pilot/';
cd(data_dir);

%% Load the Data
[Data] = read_cMEG_data_tsv(data_dir);


%% Reorder Trigger Info into Fieldtrip Data Structure
trigs                     = [];
trigs.label               = Data.Channel_Info.name(contains(Data.Channel_Info.name,'Trigger'));
trigs.fsample             = Data.samp_frequency;
trigs.time{1}             = Data.time;
trigs.trial{1}            = Data.trigger';
trigs.sampleinfo          = [1 size(trigs.trial{1},2)];

cfg             = [];
cfg.ylim        = [-0.4 0.4];
cfg.blocksize   = 30;
ft_databrowser(cfg,trigs)

%% Reorder into Fieldtrip Data Structure
rawData                     = [];
rawData.label               = Data.sensornamesinuse;
rawData.fsample             = Data.samp_frequency;
rawData.time{1}             = Data.time;
rawData.trial{1}            = Data.OPM_data;
rawData.sampleinfo          = [1 size(rawData.trial{1},2)];

% Now do the same for the header information
rawData.hdr                 = [];
rawData.hdr.label           = Data.sensornamesinuse;
find_chan = contains(Data.Channel_Info.name,Data.sensornamesinuse);
rawData.hdr.chanunit   = repmat({'fT'},size(Data.sensornamesinuse));
rawData.hdr.chantype   = Data.Channel_Info.type(find_chan);
rawData.hdr.dimord          = 'chan_time';
rawData.hdr.Fs              = Data.samp_frequency;
rawData.hdr.nSamples        = size(rawData.trial{1},2);
rawData.hdr.nTrials         = 1;
rawData.hdr.nSamplesPre     = 0;
% rawData.hdr.fieldori        = channels.fieldori;

try
    rawData.grad.chanpos    = Data.Layout_Info.Position;
    rawData.grad.chanori    = Data.Layout_Info.Orientation;
    rawData.grad.label      = Data.sensornamesinuse;
    rawData.grad.coilpos    = Data.Layout_Info.Position;
    rawData.grad.coilori    = Data.Layout_Info.Orientation;
    
    % Find the position of the channels in overall channel list
    find_chan = contains(Data.Channel_Info.name,Data.sensornamesinuse);
    rawData.grad.chanunit   = repmat({'fT'},size(Data.sensornamesinuse));
    rawData.grad.chantype   = Data.Channel_Info.type(find_chan);
    rawData.grad.tra = diag(ones(1,length(rawData.grad.label)));

catch
    disp('Could not find channel information');
end

%% Plot the Sensor Layout
figure; ft_plot_sens(grad,'label','yes','fontsize',...
    4,'facecolor','purple'); view([-90 0]);

%% Plot PSD
cfg                 = [];
cfg.trial_length    = 10;
cfg.method          = 'tim';
cfg.foi             = [0.1 150];
cfg.plot            = 'yes';
cfg.plot_chans      = 'yes';
cfg.plot_ci         = 'no';
cfg.plot_legend     = 'yes';
[pow freq]          = ft_opm_psd(cfg,rawData);
ylim([5 1e4])

%% Check for Outlier Sensors
pow_median = nanmedian(pow,3);

% Use a smaller frequency range (2-80 Hz)
freqs_for_outliers = [2 80];
freqs_include = and(freq > freqs_for_outliers(1),...
    freq < freqs_for_outliers(2));
sss = median(log10(pow_median(freqs_include,:)));

%figure; plot(freq(freqs_include),log10(pow_median(freqs_include,:)),'-k','LineWidth',1);

% % This uses GESD
% out = isoutlier(sss,'gesd','ThresholdFactor',0.999)

% This uses the median and percentiles
out = isoutlier(sss,'percentiles',[2 98]);

% Plot the good channels in colour and the bad channels in black
cfg.transparency = 0.2;
cfg.plot_mean = 'no';
cfg.foi             = [0.1 150];
cfg.plot            = 'yes';
cfg.plot_chans      = 'yes';
cfg.plot_ci         = 'no';
cfg.plot_legend     = 'no';
cfg.interactive     = 'no';
plot_powspctrm(pow_median(:,~out),cfg,rawData.label(~out),freq);
hold on; plot(freq,pow_median(:,out)','-k','LineWidth',1);
legend(rawData.label(out));

disp(['Bad Sensors:'])
rawData.label(out)

%% Remove outliers
cfg = [];
cfg.channel = rawData.label(~out);
rawData_good = ft_selectdata(cfg,rawData);

%% HFC
cfg         = [];
cfg.order   = 2;
data_hfc    = ft_denoise_hfc(cfg,rawData_good);

% Plot PSD
cfg                 = [];
cfg.trial_length    = 10;
cfg.method          = 'tim';
cfg.foi             = [0.1 150];
cfg.plot            = 'yes';
cfg.plot_chans      = 'yes';
cfg.plot_ci         = 'no';
cfg.plot_legend     = 'no';
[pow freq]          = ft_opm_psd(cfg,data_hfc);
ylim([5 1e4])
ft_opm_psd_compare(cfg,rawData_good,data_hfc);

