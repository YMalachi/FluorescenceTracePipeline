function run_model_batch_from_files(input_mat_path, output_mat_path, model_name)
% run_model_batch_from_files
%
% Goal:
%   Same idea as run_model_from_files.m, but for many ROIs from one video in
%   a single MATLAB session -- avoids paying MATLAB's several-second startup
%   cost once per ROI when running the full detected-neuron set.
%
% Usage (from Python, via subprocess, cwd = this file's folder):
%   matlab -batch "run_model_batch_from_files('in.mat','out.mat','oasis')"
%
% Input file (input_mat_path) must contain:
%   calcium_matrix - (n_rois x T) fluorescence traces, one row per ROI
%   roi_ids        - (n_rois x 1) integer ROI ids, for bookkeeping only
%   fps            - sampling rate (Hz), scalar
%   params         - struct with the fields required by the chosen model
%
% Output file (output_mat_path):
%   predicted_events   - (n_rois x T) logical
%   n_predicted_events - (n_rois x 1) double
%   roi_ids            - (n_rois x 1), echoed back for alignment
%   failed_roi_ids     - ids of ROIs whose model call errored (0 events
%                         recorded for these, NOT silently dropped)
%   failed_roi_reasons - matching error messages, same order
%
% A single ROI's model call CAN legitimately fail without aborting the
% whole batch -- e.g. run_oasis_model() throws "activity estimate has zero
% standard deviation" when OASIS's own deconvolution decides a trace has no
% detectable activity at all (observed on a small/borderline ROI once the
% detected-neuron count grew past ~30). That is a real degenerate case, not
% a bug worth crashing 65 other ROIs' results over -- it's recorded as 0
% events plus a warning, not silently swallowed. This does NOT change
% run_oasis_model.m or any other src/*.m model code -- only this batch
% wrapper's handling of a per-ROI failure.
%
% A failure BEFORE the per-ROI loop (bad input file, unknown model_name,
% mismatched array sizes) still aborts the whole run -- that's a real
% setup problem, not a per-trace edge case.

    addpath(fullfile('..', 'src'));
    addpath(genpath(fullfile('..', 'external')));

    try
        in = load(input_mat_path);

        calcium_matrix = double(in.calcium_matrix);
        roi_ids = double(in.roi_ids(:));
        fps = double(in.fps);
        params = in.params;

        n_rois = size(calcium_matrix, 1);
        n_frames = size(calcium_matrix, 2);

        if length(roi_ids) ~= n_rois
            error('roi_ids length (%d) does not match calcium_matrix rows (%d).', ...
                length(roi_ids), n_rois);
        end

        if ~ismember(model_name, {'oasis', 'simple_baseline', 'mlspike'})
            error('Unknown model_name: %s', model_name);
        end

        predicted_events = false(n_rois, n_frames);
        n_predicted_events = zeros(n_rois, 1);
        failed_roi_ids = [];
        failed_roi_reasons = {};

        for i = 1:n_rois
            calcium = calcium_matrix(i, :);

            try
                switch model_name
                    case 'oasis'
                        result = run_oasis_model(calcium, fps, params);
                    case 'simple_baseline'
                        result = run_simple_baseline(calcium, fps, params);
                    case 'mlspike'
                        result = run_mlspike_model(calcium, fps, params);
                end

                predicted_events(i, :) = logical(result.predicted_events(:))';
                n_predicted_events(i) = result.n_predicted_events;

                fprintf('ROI #%d: %d predicted events\n', roi_ids(i), n_predicted_events(i));

            catch roi_ME
                % Row stays false/0 (already initialized) -- recorded as a
                % failure, not silently treated as "0 real events".
                failed_roi_ids(end + 1) = roi_ids(i); %#ok<AGROW>
                failed_roi_reasons{end + 1} = roi_ME.message; %#ok<AGROW>
                fprintf(2, 'ROI #%d: model call FAILED -- %s (recorded as 0 events)\n', ...
                    roi_ids(i), roi_ME.message);
            end
        end

        save(output_mat_path, 'predicted_events', 'n_predicted_events', 'roi_ids', ...
            'failed_roi_ids', 'failed_roi_reasons', '-v7');

        fprintf('run_model_batch_from_files: OK, model=%s, n_rois=%d, total_events=%d, failed_rois=%d\n', ...
            model_name, n_rois, sum(n_predicted_events), length(failed_roi_ids));

    catch ME
        fprintf(2, 'run_model_batch_from_files FAILED (model=%s)\n', model_name);
        fprintf(2, '%s\n', ME.getReport('extended', 'hyperlinks', 'off'));
        rethrow(ME);
    end
end
