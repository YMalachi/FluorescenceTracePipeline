function run_model_from_files(input_mat_path, output_mat_path, model_name)
% run_model_from_files
%
% Goal:
%   Generic file-based bridge so the Python pipeline (neuron_trace_pipeline.py)
%   can invoke the existing MATLAB spike-detection models via subprocess,
%   without duplicating any model logic in Python.
%
% Usage (from Python, via subprocess, cwd = this file's folder so the
% relative addpath below resolves the same way the other scripts/*.m files
% already do):
%   matlab -batch "run_model_from_files('C:/.../input.mat','C:/.../output.mat','simple_baseline')"
%
% Input file (input_mat_path) must contain:
%   calcium - fluorescence trace vector
%   fps     - sampling rate (Hz), scalar
%   params  - struct with the fields required by the chosen model
%             (see src/run_oasis_model.m / run_simple_baseline.m / run_mlspike_model.m)
%
% Output file (output_mat_path):
%   The model's result struct, saved FLAT (one variable per field, via
%   `save(..., '-struct', result)`) so scipy.io.loadmat on the Python side
%   gets plain arrays back with no struct-unpacking needed.
%
% model_name: "oasis" | "simple_baseline" | "mlspike"
%
% Any error (including a missing external toolbox, e.g. OASIS_matlab under
% external/) is left to propagate after being logged, so `matlab -batch`
% exits non-zero and Python can detect failure from the subprocess return
% code + captured stderr rather than silently producing no output file.

    addpath(fullfile('..', 'src'));
    addpath(genpath(fullfile('..', 'external')));  % no-op (with a warning) if absent

    try
        in = load(input_mat_path);

        if ~isfield(in, 'calcium') || ~isfield(in, 'fps') || ~isfield(in, 'params')
            error('Input file must contain calcium, fps, and params.');
        end

        calcium = double(in.calcium(:));
        fps = double(in.fps);
        params = in.params;

        switch model_name
            case 'oasis'
                result = run_oasis_model(calcium, fps, params);
            case 'simple_baseline'
                result = run_simple_baseline(calcium, fps, params);
            case 'mlspike'
                result = run_mlspike_model(calcium, fps, params);
            otherwise
                error('Unknown model_name: %s', model_name);
        end

        save(output_mat_path, '-struct', 'result', '-v7');

        fprintf('run_model_from_files: OK, model=%s, output=%s\n', model_name, output_mat_path);

    catch ME
        fprintf(2, 'run_model_from_files FAILED (model=%s)\n', model_name);
        fprintf(2, '%s\n', ME.getReport('extended', 'hyperlinks', 'off'));
        rethrow(ME);
    end
end
