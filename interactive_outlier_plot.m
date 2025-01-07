function interactive_outlier_plot_existing(pow, freq, rawData, ax)
    % Initial setup
    pow_median = nanmedian(pow, 3);
    
    % Use a smaller frequency range (2-80 Hz)
    freqs_for_outliers = [2, 80];
    freqs_include = (freq > freqs_for_outliers(1)) & (freq < freqs_for_outliers(2));
    sss = median(log10(pow_median(freqs_include, :)));
    
    % Initial percentile range
    initial_percentiles = [2, 98];
    
    % Assume `ax` contains the existing graph
    hold(ax, 'on');
    
    % Plot initial black lines for bad channels (outliers)
    out = isoutlier(sss, 'percentiles', initial_percentiles);
    black_lines = gobjects(1, size(pow_median, 2)); % Preallocate line objects
    for i = 1:size(pow_median, 2)
        if out(i)
            black_lines(i) = plot(ax, freq, pow_median(:, i), '-k', 'LineWidth', 1);
        else
            black_lines(i) = plot(ax, freq, nan(size(freq)), '-k', 'LineWidth', 1); % Invisible initially
        end
    end
    
    % Add slider for lower percentile
    lower_slider = uicontrol('Style', 'slider', ...
                             'Min', 0, 'Max', 50, 'Value', initial_percentiles(1), ...
                             'Position', [150, 50, 300, 20], ...
                             'Callback', @(src, event) update_plot());
    uicontrol('Style', 'text', 'Position', [150, 70, 100, 20], ...
              'String', 'Lower Percentile');
    
    % Add slider for upper percentile
    upper_slider = uicontrol('Style', 'slider', ...
                             'Min', 50, 'Max', 100, 'Value', initial_percentiles(2), ...
                             'Position', [500, 50, 300, 20], ...
                             'Callback', @(src, event) update_plot());
    uicontrol('Style', 'text', 'Position', [500, 70, 100, 20], ...
              'String', 'Upper Percentile');
    
    % Callback function to update the black lines only
    function update_plot()
        % Get updated percentiles
        lower_p = round(lower_slider.Value);
        upper_p = round(upper_slider.Value);
        
        % Identify new outliers
        out = isoutlier(sss, 'percentiles', [lower_p, upper_p]);
        
        % Update black lines
        for i = 1:size(pow_median, 2)
            if out(i)
                black_lines(i).YData = pow_median(:, i); % Update YData for outliers
                black_lines(i).Visible = 'on';
            else
                black_lines(i).YData = nan(size(freq)); % Hide non-outliers
                black_lines(i).Visible = 'off';
            end
        end
        
        % Display outlier sensors
        disp(['Outliers for Percentiles [', num2str([lower_p, upper_p]), ']:']);
        disp(rawData.label(out));
    end
end
