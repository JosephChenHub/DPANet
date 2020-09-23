% train set
 datasets = {'NLPR', 'NJUD'};
 %datasets = {'RGBD135', 'LFSD', 'NLPR', 'NJUD', 'STEREO797', 'SSD100', 'DUT', 'SIP'};
  
 for i=1:length(datasets)
     path = char(fullfile('test/', datasets(i)));
     data = dir(char(fullfile(path, 'depth', '/*.jpg')));
     mkdir(fullfile(path, 'ostu_depth'));
     for j = 1:length(data)
        im_name = char(fullfile(path, '/depth', data(j).name));
        disp(im_name);
        img = imread(im_name);
        level = graythresh(img);
        bw = imbinarize(img, level);
        out_name = char(fullfile(path, 'ostu_depth/', data(j).name));
        disp(out_name); 
     
 %         pause;
 %         subplot(121);
 %         imshow(img);
 %         subplot(122);
 %         imshow(bw);
        imwrite(bw, out_name);
     end 
 end

