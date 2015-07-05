% Load 3D image and ground truth
load 'Pavia.mat';
% load 'Pavia_gt.mat';
ndimage = pavia;
% ground_truth = pavia_gt;

% % Visualize 3D image and ground truth
% subplot(1,2,1);
% imagesc(ndimage(:,:,50));
% subplot(1,2,2);
% imagesc(ground_truth);

height = size(ndimage, 1)/4; %reduce data size by dividing
width = size(ndimage, 2);
valsperimg = size(ndimage, 3);

% Write to .txt file
% list of values delimited by whitespace
f = fopen('pavia_centre_image_quarter4.txt', 'w');

for i = (height*3+1):(height*4)
    for j = 1:width
        for k = 1:valsperimg
            if ~((i==height*4)&&(j==width)&&(k==valsperimg))
                fprintf(f, '%d ', ndimage(i,j,k));
            else
                fprintf(f, '%d', ndimage(i,j,k));
            end
        end
    end
    
    i
end

fclose(f);

% height = size(ground_truth, 1)/4; %reduce data size by dividing
% width = size(ground_truth, 2);
% 
% % Write to .txt file
% % list of labels delimited by whitespace
% f = fopen('pavia_centre_groundtruth_quarter4.txt', 'w');
% 
% for i = (height*3+1):(height*4)
%     for j = 1:(width)
%         if ~((i==height*4)&&(j==width))
%             fprintf(f, '%d ', ground_truth(i,j));
%         else
%             fprintf(f, '%d', ground_truth(i,j));
%         end
%     end
%     
%     i
% end
% 
% fclose(f);