function headshape_to_ply(headshape, output_plyfile)
    % Load the headshape data from FieldTrip .mat file
    
    % Assuming 'headshape' contains the 'pos' field with 3D coordinates
    pos = headshape.pos;  % Nx3 matrix of 3D coordinates
    
    % Optionally: add color data (for simplicity, we'll use random colors)
    % Assuming you have a field in the headshape with color or using default
    try
        colors = headshape.color;
    catch 
        colors = rand(size(pos, 1), 3);  % Random RGB colors for each point
    end
    
    % Assuming headshape contains 'tri' field with triangle faces as indices
    % 'tri' should be an Mx3 matrix where each row contains indices of 3 vertices forming a triangle
    if isfield(headshape, 'tri')
        faces = headshape.tri;  % Mx3 matrix of triangle face indices
    else
        error('No triangle data found in the headshape structure.');
    end
    
    % Create the header for the PLY file
    ply_header = ['ply\n', ...
                  'format ascii 1.0\n', ...
                  'element vertex ', num2str(size(pos, 1)), '\n', ...
                  'property float x\n', ...
                  'property float y\n', ...
                  'property float z\n', ...
                  'property uchar red\n', ...
                  'property uchar green\n', ...
                  'property uchar blue\n', ...
                  'element face ', num2str(size(faces, 1)), '\n', ...
                  'property list uchar int vertex_indices\n', ...
                  'end_header\n'];
    
    % Open the output PLY file
    fid = fopen(output_plyfile, 'w');
    
    % Write the header to the file
    fprintf(fid, ply_header);
    
    % Write the vertex data (position + color)
    for i = 1:size(pos, 1)
        % Ensure colors are in the range [0, 255]
        r = round(colors(i, 1) * 255);
        g = round(colors(i, 2) * 255);
        b = round(colors(i, 3) * 255);
        
        % Write the vertex data (position + color)
        fprintf(fid, '%f %f %f %d %d %d\n', pos(i, 1), pos(i, 2), pos(i, 3), r, g, b);
    end
    
    % Write the face data (indices of vertices forming each triangle)
    for i = 1:size(faces, 1)
        % PLY uses 0-based indexing, so we need to subtract 1 from the indices
        fprintf(fid, '3 %d %d %d\n', faces(i, 1) - 1, faces(i, 2) - 1, faces(i, 3) - 1);
    end
    
    % Close the file
    fclose(fid);
    
    disp(['PLY file with vertices and faces saved as: ', output_plyfile]);
end
