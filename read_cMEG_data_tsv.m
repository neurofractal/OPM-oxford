function [Data] = read_cMEG_data_tsv(folder_name)
% Basic read in of raw data with Channel names and locations.
% Also reads and applies the gain conversion to put data into fT (not
% including triggers). Raw data is recorded as Data.data_Volts

% Get all cMEG files
cMEG_files = dir(fullfile(folder_name, '*.cMEG'));
for n = 1:size(cMEG_files,1)
    comb_file_id(n) = contains(cMEG_files(n).name,'_meg.cMEG');
end
file_index = find(comb_file_id);

% If there is no combined file, do that
if isempty(file_index)
    disp('Combining cMEG files')
    for n = 1:size(cMEG_files,1)
        all_filenames{n} = fullfile(cMEG_files(n).folder,cMEG_files(n).name);
    end
    filename = cMEG_combine_nottsapp(all_filenames);
else
    filename = fullfile(cMEG_files(file_index).folder,cMEG_files(file_index).name);
end

if strcmpi(num2str(filename(end-4:end)),'.cMEG')
    filename = filename(1:end-5);
    [~,OPM_vars.data_filename] = fileparts(filename);
end

disp('Get data')
fname = [filename '.cMEG'];
fid = fopen(fname,'rb','ieee-be');
finfo = dir(fname);
fsize = finfo.bytes;
Adim_conv = [2^32; 2^16; 2^8; 1]; % Array dimension conversion table

disp('Preallocation')
I = 0;
while fsize ~= ftell(fid)
    %                     disp(I)
    dims = [];
    for n = 1:2 % 2 dimensions (Nch x Time)
        temp = fread(fid,4);
        temp = sum(Adim_conv.*temp);
        dims = [dims,temp];
    end
    I = I + 1;
    temp = fread(fid,prod(dims),'double',0,'ieee-be');  % Skip the actual data (much faster for some reason)
end
fseek(fid,0,'bof'); % Reset the cursor
data1 = repmat({NaN*ones(dims)},I,1);  % Preallocate space, assumes each section is the same

disp('Load and parse data')
for j = 1:I
    dims = [];  % Actual array dimensions
    for n = 1:2 % 2 dimensions (Nch x Time)
        temp = fread(fid,4);
        temp = sum(Adim_conv.*temp);
        dims = [dims,temp];
    end
    clear temp n
    temp = fread(fid,prod(dims),'double',0,'ieee-be');

    % Reshape the data into the correct array configuration
    for n = 1:2
        temp = reshape(temp,dims(2-n+1),[])';
    end
    data1{j} = temp;  % Save the data
end
fclose(fid);  % Close the file

%             disp('Reshape into sensible order')
data = NaN*zeros(size(data1{1},2),size(data1{1},1).*size(data1,1));
for n = 1:size(data1,1)
    data_range = [(n-1)*size(data1{1},1)+1:n*size(data1{1},1)];
    data(:,data_range) = [data1{n}']; % data in Nch+triggers+1 x time, but channel one is the time
end

Data.samp_frequency = round(1/(data(1,2)-data(1,1)));
Data.nsamples = size(data,2);
Data.time = linspace(0,size(data,2)./Data.samp_frequency,size(data,2));
Data.elapsed_time = Data.nsamples./Data.samp_frequency;
Data.data_Volts = data(2:end,:); % Data in volts

disp('Getting Channel Information')
try
    channel_info_file = dir(fullfile(folder_name, '*_channels.tsv'));
    Data.Channel_Info = tdfread(fullfile(channel_info_file.folder,channel_info_file.name));
    CI_fields = fields(Data.Channel_Info);
    for n = 1:size(CI_fields,1)
        try
            Data.Channel_Info.(CI_fields{n}) = cellstr(Data.Channel_Info.(CI_fields{n}));
        end
    end

    % Apply gain conversion
    gain_conv_vals = [];

    for n = size(Data.Channel_Info.(CI_fields{end})):-1:1
        try
            gain_conv_vals(n,1) = str2num(Data.Channel_Info.(CI_fields{end}){n})./1e6;
        catch
            gain_conv_vals(n,1) = 1;
        end
    end

    Data.data = Data.data_Volts./gain_conv_vals; % Data in fT (not inc. Triggers)

catch
    disp('Issue with Channel Information, skipping for now')
end

disp('Getting Session Information')
try
    sesh_info_file = dir([folder_name '\*_meg.json']);
    sesh_fid = fopen([sesh_info_file.folder '\' sesh_info_file.name]);
    raw = fread(sesh_fid,inf);
    str = char(raw');
    fclose(sesh_fid);
    Data.Session_Info = jsondecode(str);
catch
    disp('Issue with Session Information, skipping for now')
end

disp('Getting Layout Information')
try
    layout_info_file = dir(fullfile(folder_name, '*_HelmConfig.tsv'));
    Layout_Info = tdfread(fullfile(layout_info_file.folder,layout_info_file.name));
    last_line_idx = strmatch('Helmet:',Layout_Info.Name);
    OPM_vars.layout_name = strtrim(Layout_Info.Px(last_line_idx,:));
    field_names = fieldnames(Layout_Info);
    for n = 1:size(field_names,1)
        Layout_Info.(field_names{n})(last_line_idx,:) = [];
        if max(strcmpi(field_names{n},{'Px';'Py';'Pz';'Ox';'Oy';'Oz';'Layx';'Layz'}))
            try
                Layout_Info.(field_names{n}) = str2num(Layout_Info.(field_names{n}));
            end
        end
    end
    Layout_Info.Sensor = cellstr(Layout_Info.Sensor);
    Layout_Info.Sensor(strcmpi(Layout_Info.Sensor,'none')) = {'[]'};
    Data.Layout_Info.SlotSensorPairs = [cellstr(Layout_Info.Name) cellstr(Layout_Info.Sensor)];
    CI_name = Data.Channel_Info.name;
    for n = 1:size(CI_name,1)
        CI_name{n} = CI_name{n}(isstrprop(CI_name{n},'alphanum'));
    end
    for n = 1:size(Data.Layout_Info.SlotSensorPairs,1)
        sens_name = Data.Layout_Info.SlotSensorPairs{n,2};
        sens_name = sens_name(isstrprop(sens_name,'alphanum'));

        data_idx = find(strcmpi(sens_name,CI_name));
        if isempty(data_idx)
            Data.Layout_Info.SlotSensorPairs{n,3} = 0;
        else
            Data.Layout_Info.SlotSensorPairs{n,3} = data_idx;
        end
    end

    Data.Layout_Info.Position = [Layout_Info.Px,Layout_Info.Py,Layout_Info.Pz];
    Data.Layout_Info.Orientation = [Layout_Info.Ox,Layout_Info.Oy,Layout_Info.Oz];
    Data.Layout_Info.Lay = [Layout_Info.Layx,Layout_Info.Layy];

catch
    disp('Issue with Layout Information, skipping for now')
end

disp('Read sensor transform file')
try
    sensor_transform_info_file = dir(fullfile(folder_name, '*_SensorTransform.tsv'));
    % Set up the Import Options and import the data
    opts = delimitedTextImportOptions("NumVariables", 4);

    % Specify range and delimiter
    opts.DataLines = [1, Inf];
    opts.Delimiter = "\t";

    % Specify column names and types
    opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4"];
    opts.VariableTypes = ["double", "double", "double", "double"];

    % Specify file level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";

    % Import the data
    SensorTransform = readtable(fullfile(sensor_transform_info_file.folder,sensor_transform_info_file.name), opts);

    % Convert to output typefilename(1:end-4)
    Data.SensorTransform = table2array(SensorTransform);

    % Clear temporary variables
    clear opts

    Data.Layout_Info.Position_crg = [Data.SensorTransform(1:3,1:3)*Data.Layout_Info.Position' + Data.SensorTransform(1:3,4)]';
    Data.Layout_Info.Orientation_crg = [Data.SensorTransform(1:3,1:3)*Data.Layout_Info.Orientation']';
    skip_transform = 1;
catch
    disp('Issue with sensor transform file, skipping for now')
    skip_transform = 0;
end


% separate triggers
Data.trigger = Data.data(strcmpi(Data.Channel_Info.type,'TRIG'),:)';
Data.BNC = Data.data(strcmpi(Data.Channel_Info.type,'MISC'),:)';
% separate OPM data and put in helmet order
if skip_transform
    disp('Reorder data to match sensor layout')
    pos_no = nonzeros(cell2mat(Data.Layout_Info.SlotSensorPairs(:,3)));
    Data.OPM_data = Data.data(pos_no,:);
    Data.sensornamesinuse = Data.Channel_Info.name(pos_no,:);


    for n = 1:length(Data.sensornamesinuse)
        ax_temp = strsplit(Data.sensornamesinuse{n},{' ','\t'});
        Data.axis{n,1} = ax_temp{2}(2)';
    end
end
disp('Done!');
end

function [fname_wt] = cMEG_combine_nottsapp(all_files)
%% Combine .cMEG files in order of all_files variables.
% all_files is an nx1 cell array of file locations
% Read Data
str_split = split(all_files{1},'_meg_');
filename_base = [str_split{1} '_meg'];
data_all = [];
for num_fname = 1:length(all_files)
    clearvars -except filename_base all_files num_fname data_all
    file_prog_str = ['Reading file ' num2str(num_fname) '/' num2str(length(all_files))];
    fname = [all_files{num_fname}];
    fid = fopen(fname,'rb','ieee-be');
    finfo = dir(fname);
    fsize = finfo.bytes;
    Adim_conv = [2^32; 2^16; 2^8; 1]; % Array dimension conversion table

    disp([file_prog_str ': Preallocation'])
    I = 0;
    while fsize ~= ftell(fid)
        %                     disp(I)
        dims = [];
        for n = 1:2 % 2 dimensions (Nch x Time)
            temp = fread(fid,4);
            temp = sum(Adim_conv.*temp);
            dims = [dims,temp];
        end
        I = I + 1;
        temp = fread(fid,prod(dims),'double',0,'ieee-be');  % Skip the actual data (much faster for some reason)
    end
    fseek(fid,0,'bof'); % Reset the cursor
    data1 = repmat({NaN*ones(dims)},I,1);  % Preallocate space, assumes each section is the same

    disp([file_prog_str ': Load and parse data'])
    for j = 1:I
        dims = [];  % Actual array dimensions
        for n = 1:2 % 2 dimensions (Nch x Time)
            temp = fread(fid,4);
            temp = sum(Adim_conv.*temp);
            dims = [dims,temp];
        end
        clear temp n
        temp = fread(fid,prod(dims),'double',0,'ieee-be');

        % Reshape the data into the correct array configuration
        %         for n = 1:2
        %             temp = reshape(temp,dims(2-n+1),[]);
        %         end
        data1{j} = temp;  % Save the data
    end
    fclose(fid);  % Close the file

    data_all = [data_all;data1];
end

Nch = dims(1);
Nsamp_seg = dims(2);

% Write combined data
fname_wt = [filename_base '.cMEG'];
disp('Writing new file to ')
disp(fname_wt)
fid = fopen(fname_wt,'w','ieee-be');
for n_data = 1:size(data_all,1)
    rder = Nch;
    rder2 = Nsamp_seg;
    for nn = 1:length(Adim_conv)
        val = floor(rder./Adim_conv(nn));
        rder = rder - Adim_conv(nn).*val;
        Nch_vec(nn) = val;

        val2 = floor(rder2./Adim_conv(nn));
        rder2 = rder2 - Adim_conv(nn).*val2;
        Nsamp_vec(nn) = val2;

    end

    hdr = [Nch_vec, Nsamp_vec];
    % Write header
    fwrite(fid,hdr);
    % Write data segment
    fwrite(fid,data_all{n_data},'double');
end
fclose(fid);
end