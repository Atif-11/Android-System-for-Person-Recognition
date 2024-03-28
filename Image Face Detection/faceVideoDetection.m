% Read the video using VideoReader function
the_Video = VideoReader("Atif.mp4");

% As videos are simply fast changing of still frames, extract the frames
video_Frame = readFrame(the_Video); 

% As done in Images, we will again use the CascadeObjectDetector function
face_Detector = vision.CascadeObjectDetector();
location_of_the_Face = step(face_Detector, video_Frame);

% Plot a rectangle around the face
detected_Frame = insertShape(video_Frame,'Rectangle', location_of_the_Face);

% Now we are going to track the faces as well, so to move the rectangle 
% around the face break rectangle into points
if ~isempty(location_of_the_Face)
    rectangle_to_Points = bbox2points(location_of_the_Face(1,:));
else
    disp('No face detected.');
    return;
end

% Extract the features of face but it only takes grayscale images
feature_Points = detectMinEigenFeatures(rgb2gray(detected_Frame), 'ROI',location_of_the_Face);
%ROI = Region of Interest

% Now we need a point tracker to point to the location of the face
pointTracker = vision.PointTracker("MaxBidirectionalError",2);

% Now convert the feature_points to x,y coordinate
feature_Points = feature_Points.Location;
% Now, let's initialize the Point Tracker
initialize(pointTracker, feature_Points, video_Frame);

% Finally, let's play the video
left = 100;
bottom = 100;
width = size(detected_Frame, 2);
height = size(detected_Frame, 1);
video_Player = vision.VideoPlayer('Position', [left bottom width height]);
% For tracking of face, we need to compare the change of the location of
% feature points
previous_Points = feature_Points;

while hasFrame(the_Video)
    
    video_Frame = readFrame(the_Video);
    [feature_Points, isFound] = step(pointTracker, video_Frame);
    % Only the first row of feature points is to be stored in new points
    new_Points = feature_Points(isFound, :);
    old_Points = previous_Points(isFound, :);

    if size(new_Points, 1) >= 2
       [transformed_Rectangle, old_Points, new_Points] = ...
           estimateGeometricTransform(old_Points, new_Points,...
           'similarity', 'MaxDistance', 4);
       rectangle_to_Points = transformPointsForward(transformed_Rectangle, rectangle_to_Points);

       reshaped_Rectangle = reshape(rectangle_to_Points', 1, []);
       detected_Frame = insertShape(video_Frame, 'Polygon', reshaped_Rectangle, 'LineWidth',2);

       detected_Frame = insertMarker(detected_Frame, new_Points, '+', 'Color', 'White');

       previous_Points = new_Points;
       setPoints(pointTracker, previous_Points);
    end
    step(video_Player, detected_Frame)
end

release(video_Player);
