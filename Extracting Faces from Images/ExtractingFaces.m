% Path to the directory where original images are stored
originalImagePath = 'Matthew Perry/Matthew Perry Original Images/';

% Path to the directory where cropped face images will be saved
croppedImagePath = 'Matthew Perry/Matthew Perry Cropped Images/';

% Load the face detector
faceDetector = vision.CascadeObjectDetector();

% Get a list of image files in the original images folder
imageFiles = dir(fullfile(originalImagePath, '*.jpg'));

% Counter for naming the cropped images
count = 1;

% Loop through each image file
for i = 1:length(imageFiles)
    % Read the image
    img = imread(fullfile(originalImagePath, imageFiles(i).name));
    
    % Convert the image to grayscale
    gray = rgb2gray(img);
    
    % Detect faces in the image
    bbox = step(faceDetector, gray);
    
    % Loop through the detected faces
    for j = 1:size(bbox, 1)
        % Extract the coordinates of the face bounding box
        x = bbox(j, 1);
        y = bbox(j, 2);
        w = bbox(j, 3);
        h = bbox(j, 4);
        
        % Crop the face region from the image
        faceROI = img(y:y+h, x:x+w, :);
        
        % Save the cropped face image
        imwrite(faceROI, fullfile(croppedImagePath, ['Matthew Face ' num2str(count) '.jpg']));
        
        % Increment the counter
        count = count + 1;
    end
end
