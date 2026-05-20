%% CREATE_DATA_SPLIT.m
% Goal:
%   Create a train / validation / test split for the calcium spike inference project.
%
% Important:
%   We split by biological unit, not by result rows.
%
% Unit definition:
%   If a recording has cell_num:
%       unit_id = dataset_id + cell_num
%
%   If a recording does not have cell_num:
%       unit_id = dataset_id + recording_idx
%
% Why:
%   Some datasets contain multiple segments from the same cell.
%   Those segments must not be split across train/validation/test.

clear; clc; close all;

%% Paths

data_dir = fullfile('..', 'data', 'MockData');
output_dir = fullfile('..', 'results', 'tables');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

output_file = fullfile(output_dir, 'data_split.csv');

%% Split settings

% Fixed random seed for reproducibility.
rng(42);

train_fraction = 0.60;
validation_fraction = 0.20;
test_fraction = 0.20;

if abs(train_fraction + validation_fraction + test_fraction - 1) > 1e-10
    error('Train/validation/test fractions must sum to 1.');
end

%% Find dataset files

dataset_files = dir(fullfile(data_dir, 'data.*.train.preprocessed.mat'));

if isempty(dataset_files)
    error('No dataset files found in: %s', data_dir);
end

fprintf('Found %d dataset files.\n', length(dataset_files));

%% Build recording table

recording_table = table();

for file_idx = 1:length(dataset_files)

    dataset_name = dataset_files(file_idx).name;
    dataset_path = fullfile(dataset_files(file_idx).folder, dataset_name);
    dataset_id = parse_dataset_id(dataset_name);

    fprintf('\nReading dataset file: %s\n', dataset_name);

    raw = load(dataset_path);

    if ~isfield(raw, 'data')
        error('File %s does not contain variable named "data".', dataset_name);
    end

    data = raw.data;

    for recording_idx = 1:numel(data)

        rec = data{recording_idx};

        if isfield(rec, 'cell_num')
            cell_num = double(rec.cell_num);
            unit_id = sprintf('dataset_%d_cell_%g', dataset_id, cell_num);
            unit_type = "cell_num";
        else
            cell_num = NaN;
            unit_id = sprintf('dataset_%d_recording_%d', dataset_id, recording_idx);
            unit_type = "recording_idx";
        end

        n_samples = length(rec.calcium(:));
        fps = double(rec.fps);
        duration_sec = n_samples / fps;
        n_spikes = sum(double(rec.spikes(:)));

        new_row = table( ...
            string(dataset_name), ...
            dataset_id, ...
            recording_idx, ...
            cell_num, ...
            string(unit_id), ...
            unit_type, ...
            n_samples, ...
            fps, ...
            duration_sec, ...
            n_spikes, ...
            'VariableNames', { ...
                'dataset_name', ...
                'dataset_id', ...
                'recording_idx', ...
                'cell_num', ...
                'unit_id', ...
                'unit_type', ...
                'n_samples', ...
                'fps', ...
                'duration_sec', ...
                'n_spikes'});

        recording_table = [recording_table; new_row];
    end
end

fprintf('\nTotal recordings found: %d\n', height(recording_table));
fprintf('Total unique units found: %d\n', length(unique(recording_table.unit_id)));

%% Split unique units within each dataset

split_table = recording_table;
split_table.split = strings(height(split_table), 1);

dataset_ids = unique(recording_table.dataset_id);

for d = 1:length(dataset_ids)

    dataset_id = dataset_ids(d);

    dataset_rows = recording_table(recording_table.dataset_id == dataset_id, :);
    unique_units = unique(dataset_rows.unit_id);

    n_units = length(unique_units);

    fprintf('\nDataset %d: %d recordings, %d unique units\n', ...
        dataset_id, height(dataset_rows), n_units);

    % Randomize unit order.
    shuffled_units = unique_units(randperm(n_units));

    % Decide split counts.
    [n_train, n_val, n_test] = choose_split_counts( ...
        n_units, ...
        train_fraction, ...
        validation_fraction, ...
        test_fraction);

    train_units = shuffled_units(1:n_train);
    val_units = shuffled_units(n_train+1:n_train+n_val);
    test_units = shuffled_units(n_train+n_val+1:n_train+n_val+n_test);

    fprintf('  Train units:      %d\n', length(train_units));
    fprintf('  Validation units: %d\n', length(val_units));
    fprintf('  Test units:       %d\n', length(test_units));

    % Assign split labels to every recording that belongs to each unit.
    split_table.split( ...
        split_table.dataset_id == dataset_id & ...
        ismember(split_table.unit_id, train_units)) = "train";

    split_table.split( ...
        split_table.dataset_id == dataset_id & ...
        ismember(split_table.unit_id, val_units)) = "validation";

    split_table.split( ...
        split_table.dataset_id == dataset_id & ...
        ismember(split_table.unit_id, test_units)) = "test";
end

%% Final sanity checks

if any(split_table.split == "")
    error('Some recordings were not assigned to any split.');
end

% Make sure each unit belongs to one split only.
all_units = unique(split_table.unit_id);

for i = 1:length(all_units)

    unit_id = all_units(i);
    unit_rows = split_table(split_table.unit_id == unit_id, :);
    unit_splits = unique(unit_rows.split);

    if length(unit_splits) ~= 1
        error('Unit %s appears in multiple splits.', unit_id);
    end
end

%% Print split summary

fprintf('\nFinal recording-level split summary:\n');
fprintf('-----------------------------------\n');
fprintf('Train recordings:      %d\n', sum(split_table.split == "train"));
fprintf('Validation recordings: %d\n', sum(split_table.split == "validation"));
fprintf('Test recordings:       %d\n', sum(split_table.split == "test"));

fprintf('\nFinal unit-level split summary:\n');
fprintf('-------------------------------\n');

train_units = unique(split_table.unit_id(split_table.split == "train"));
val_units = unique(split_table.unit_id(split_table.split == "validation"));
test_units = unique(split_table.unit_id(split_table.split == "test"));

fprintf('Train units:      %d\n', length(train_units));
fprintf('Validation units: %d\n', length(val_units));
fprintf('Test units:       %d\n', length(test_units));

%% Save split table

writetable(split_table, output_file);

fprintf('\nSaved split table to:\n%s\n', output_file);

%% Local helper functions

function dataset_id = parse_dataset_id(dataset_name)
% parse_dataset_id
%
% Extracts dataset number from names like:
%   data.1.train.preprocessed.mat

    tokens = regexp(dataset_name, 'data\.(\d+)\.train', 'tokens');

    if isempty(tokens)
        dataset_id = NaN;
    else
        dataset_id = str2double(tokens{1}{1});
    end
end

function [n_train, n_val, n_test] = choose_split_counts(n_units, train_fraction, validation_fraction, test_fraction)
% choose_split_counts
%
% Chooses train/validation/test counts for one dataset.
%
% For very small datasets, we still try to assign at least one validation
% and one test unit when possible.

    if n_units <= 0
        error('n_units must be positive.');
    end

    if n_units == 1
        % With only one unit, it must go to train.
        n_train = 1;
        n_val = 0;
        n_test = 0;
        return;
    end

    if n_units == 2
        % With two units, use one train and one test.
        % Validation is not possible without making train empty.
        n_train = 1;
        n_val = 0;
        n_test = 1;
        return;
    end

    % Initial rounded counts.
    n_train = round(train_fraction * n_units);
    n_val = round(validation_fraction * n_units);
    n_test = n_units - n_train - n_val;

    % Guarantee at least one validation and one test unit when possible.
    if n_val < 1
        n_val = 1;
    end

    if n_test < 1
        n_test = 1;
    end

    % Adjust train count so total matches n_units.
    n_train = n_units - n_val - n_test;

    % If rounding made train too small, force at least one train unit.
    if n_train < 1
        n_train = 1;

        % Remove from the larger of validation/test.
        if n_val >= n_test && n_val > 1
            n_val = n_val - 1;
        elseif n_test > 1
            n_test = n_test - 1;
        else
            error('Could not create a valid split for n_units = %d.', n_units);
        end
    end

    % Final safety check.
    if n_train + n_val + n_test ~= n_units
        error('Split counts do not sum to n_units.');
    end
end