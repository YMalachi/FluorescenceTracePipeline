function metrics = evaluate_event_prediction(predicted_events, spikes, fps, tolerance_sec, mode, burst_gap_sec)
% evaluate_event_prediction
%
% Goal:
%   Evaluate predicted event times against ground-truth binned spikes.
%
% Evaluation modes:
%
%   mode = 'spike_bins'
%       Every bin with at least one spike is treated as a true event.
%
%   mode = 'burst_onsets'
%       Nearby spike bins are grouped into bursts.
%       Only the first bin of each burst is treated as the true event.
%
% Matching logic:
%   A predicted event is counted as correct if it falls within
%   +/- tolerance_sec of an unmatched true event.
%
% Important:
%   Each predicted event can match at most one true event.
%   Each true event can match at most one predicted event.
%
% Inputs:
%   predicted_events - logical/numeric vector, 1 where model predicts event
%   spikes           - binned spike train, same length as predicted_events
%   fps              - sampling rate in Hz
%   tolerance_sec    - matching tolerance in seconds
%   mode             - 'spike_bins' or 'burst_onsets'
%   burst_gap_sec    - max gap between spike bins to consider same burst
%
% Output:
%   metrics - struct with detection and timing metrics

    %% Basic input checks

    if nargin < 4
        error('evaluate_event_prediction requires at least 4 inputs: predicted_events, spikes, fps, tolerance_sec.');
    end

    if nargin < 5 || isempty(mode)
        mode = 'spike_bins';
    end

    if nargin < 6 || isempty(burst_gap_sec)
        burst_gap_sec = 0.10;
    end

    if isempty(predicted_events)
        error('predicted_events is empty.');
    end

    if isempty(spikes)
        error('spikes is empty.');
    end

    if isempty(fps) || ~isnumeric(fps) || ~isscalar(fps)
        error('fps must be a single numeric value.');
    end

    if fps <= 0
        error('fps must be positive. Received: %.4f', fps);
    end

    if isempty(tolerance_sec) || ~isnumeric(tolerance_sec) || ~isscalar(tolerance_sec)
        error('tolerance_sec must be a single numeric value.');
    end

    if tolerance_sec < 0
        error('tolerance_sec must be non-negative. Received: %.4f', tolerance_sec);
    end

    if isempty(burst_gap_sec) || ~isnumeric(burst_gap_sec) || ~isscalar(burst_gap_sec)
        error('burst_gap_sec must be a single numeric value.');
    end

    if burst_gap_sec < 0
        error('burst_gap_sec must be non-negative. Received: %.4f', burst_gap_sec);
    end

    %% Convert inputs to column vectors

    predicted_events = logical(predicted_events(:));
    spikes = double(spikes(:));

    if length(predicted_events) ~= length(spikes)
        error(['predicted_events and spikes must have the same length. ', ...
               'predicted_events length = %d, spikes length = %d.'], ...
               length(predicted_events), length(spikes));
    end

    if any(isnan(spikes)) || any(isinf(spikes))
        error('spikes contains NaN or Inf values.');
    end

    if any(spikes < 0)
        error('spikes contains negative values, which is invalid for spike counts.');
    end

    if any(abs(spikes - round(spikes)) > 1e-10)
        error('spikes contains non-integer values. Expected binned spike counts.');
    end

    %% Build ground-truth event vector

    switch mode

        case 'spike_bins'

            % Every bin with at least one spike is a true event.
            true_events = spikes > 0;

        case 'burst_onsets'

            % First, find all bins with spikes.
            spike_bins = find(spikes > 0);

            true_events = false(size(spikes));

            if ~isempty(spike_bins)

                % Convert burst gap from seconds to bins.
                burst_gap_bins = round(burst_gap_sec * fps);

                % The first spike bin always starts a burst.
                true_events(spike_bins(1)) = true;

                % A new burst starts when the gap from the previous spike bin
                % is larger than burst_gap_bins.
                for i = 2:length(spike_bins)

                    gap_from_previous_spike_bin = spike_bins(i) - spike_bins(i-1);

                    if gap_from_previous_spike_bin > burst_gap_bins
                        true_events(spike_bins(i)) = true;
                    end
                end
            end

        otherwise

            error('Unknown evaluation mode: %s. Use ''spike_bins'' or ''burst_onsets''.', mode);
    end

    pred_idx = find(predicted_events);
    true_idx = find(true_events);

    n_predicted_events = length(pred_idx);
    n_true_events = length(true_idx);

    %% Convert tolerance from seconds to bins

    tolerance_bins = round(tolerance_sec * fps);

    %% Match predicted events to true events

    % matched_true keeps track of which true events were already matched.
    matched_true = false(size(true_idx));

    matched_pred_idx = [];
    matched_true_idx = [];
    timing_errors_sec = [];

    FP = 0;

    for p = 1:n_predicted_events

        current_pred_idx = pred_idx(p);

        % Find unmatched true events only.
        available_true_idx = true_idx(~matched_true);

        if isempty(available_true_idx)
            FP = FP + 1;
            continue;
        end

        % Distance in bins between current prediction and all unmatched true events.
        distances_bins = abs(available_true_idx - current_pred_idx);

        % Find nearest true event.
        [min_distance_bins, nearest_local_idx] = min(distances_bins);

        if min_distance_bins <= tolerance_bins

            % Convert local index among available true events back to index in true_idx.
            available_positions = find(~matched_true);
            nearest_global_position = available_positions(nearest_local_idx);

            % Mark this true event as matched.
            matched_true(nearest_global_position) = true;

            matched_pred_idx(end+1, 1) = current_pred_idx;
            matched_true_idx(end+1, 1) = true_idx(nearest_global_position);

            % Signed timing error:
            % positive = prediction occurred after true event
            % negative = prediction occurred before true event
            timing_errors_sec(end+1, 1) = ...
                (current_pred_idx - true_idx(nearest_global_position)) / fps;

        else
            FP = FP + 1;
        end
    end

    TP = length(matched_pred_idx);
    FN = sum(~matched_true);

    %% Detection metrics

    if TP + FP == 0
        precision = NaN;
    else
        precision = TP / (TP + FP);
    end

    if TP + FN == 0
        recall = NaN;
    else
        recall = TP / (TP + FN);
    end

    if isnan(precision) || isnan(recall) || (precision + recall == 0)
        F1 = NaN;
    else
        F1 = 2 * precision * recall / (precision + recall);
    end

    %% Timing metrics

    if isempty(timing_errors_sec)
        mean_signed_timing_error_sec = NaN;
        median_signed_timing_error_sec = NaN;
        mean_abs_timing_error_sec = NaN;
        median_abs_timing_error_sec = NaN;
        std_timing_error_sec = NaN;
    else
        mean_signed_timing_error_sec = mean(timing_errors_sec);
        median_signed_timing_error_sec = median(timing_errors_sec);
        mean_abs_timing_error_sec = mean(abs(timing_errors_sec));
        median_abs_timing_error_sec = median(abs(timing_errors_sec));
        std_timing_error_sec = std(timing_errors_sec);
    end

    %% Package output

    metrics = struct();

    % Evaluation mode
    metrics.mode = mode;
    metrics.burst_gap_sec = burst_gap_sec;

    % Event counts
    metrics.n_predicted_events = n_predicted_events;
    metrics.n_true_events = n_true_events;
    metrics.n_matched_events = TP;

    % Confusion counts
    metrics.TP = TP;
    metrics.FP = FP;
    metrics.FN = FN;

    % Detection metrics
    metrics.precision = precision;
    metrics.recall = recall;
    metrics.F1 = F1;

    % Timing metrics
    metrics.timing_errors_sec = timing_errors_sec;
    metrics.mean_signed_timing_error_sec = mean_signed_timing_error_sec;
    metrics.median_signed_timing_error_sec = median_signed_timing_error_sec;
    metrics.mean_abs_timing_error_sec = mean_abs_timing_error_sec;
    metrics.median_abs_timing_error_sec = median_abs_timing_error_sec;
    metrics.std_timing_error_sec = std_timing_error_sec;

    % Matching details
    metrics.matched_pred_idx = matched_pred_idx;
    metrics.matched_true_idx = matched_true_idx;

    % Evaluation settings
    metrics.fps = fps;
    metrics.tolerance_sec = tolerance_sec;
    metrics.tolerance_bins = tolerance_bins;

    % Ground-truth vector used internally.
    % This is useful for plotting/debugging later.
    metrics.true_events = true_events;
end