function circumference = estimate_circumference(headshape, z_level)
    % Function to estimate the circumference of a head model slice at a given z-level
    %
    % Inputs:
    %   headshape - Headshape structure containing vertices and faces
    %   z_level - The height (z-value) at which to slice the model
    %
    % Output:
    %   circumference - Estimated circumference of the head model at z_level

    % Plot the realigned headshape
    figure; hold on;
    ft_plot_headshape(headshape);
    ft_plot_axes(headshape);
    view([125 10]);
    lighting gouraud;
    material dull;
    light;

    % Extract vertices and faces from the headshape
    vertices = headshape.pos;
    faces = headshape.tri;

    % Define tolerance for intersection calculations
    tol = 1e-3;

    % Find edges of the model
    edges = unique(sort([faces(:, [1, 2]); faces(:, [2, 3]); faces(:, [1, 3])], 2), 'rows');

    % Get vertices of the edges
    v1 = vertices(edges(:, 1), :);
    v2 = vertices(edges(:, 2), :);

    % Find edges that intersect the plane
    mask = (v1(:, 3) - z_level) .* (v2(:, 3) - z_level) < 0; % Sign change
    intersecting_edges = edges(mask, :);

    % Interpolate to find intersection points
    t = (z_level - v1(mask, 3)) ./ (v2(mask, 3) - v1(mask, 3));
    intersection_points = v1(mask, :) + t .* (v2(mask, :) - v1(mask, :));

    % Plot the intersection points on the 3D headshape
    figure;
    set(gcf, 'Position', [1 1 1600 800]);
    subplot(1, 2, 1);
    hold on;
    ft_plot_headshape(headshape);
    ft_plot_axes(headshape);
    ft_plot_mesh(intersection_points, 'vertexcolor', 'b', 'vertexsize', 20);
    view([125 10]);
    lighting gouraud;
    material dull;
    light;

    % Project intersection points to the 2D xy-plane
    projected_points = intersection_points(:, 1:2);

    % Use boundary to order the points into a closed loop
    shrink_factor = 0.9; % Adjust between 0 (convex hull) and 1 (tighter fit)
    k = boundary(projected_points(:, 1), projected_points(:, 2), shrink_factor);

    % Extract ordered points
    ordered_points = projected_points(k, :);

    % Visualize the boundary to verify correctness
    subplot(1, 2, 2);
    plot(ordered_points(:, 1), ordered_points(:, 2), 'k-', 'LineWidth', 4);
    hold on;
    scatter(projected_points(:, 1), projected_points(:, 2), 10, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
    axis equal;
    xlabel('X (mm)');
    ylabel('Y (mm)');

    % Compute the circumference
    distances = sqrt(sum(diff([ordered_points; ordered_points(1, :)]).^2, 2));
    circumference = sum(distances);

    % Add the circumference to the plot title
    title(sprintf('Estimated Circumference (using boundary): %.2f mm', circumference));

    % Display the result in the command window
    fprintf('Estimated Circumference (using boundary): %.2f mm\n', circumference);
end
