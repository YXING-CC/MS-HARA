clc;clear

load label_dict.mat
load label_verb.mat
load label_noun.mat

% transit_lab = 73:79;
% transit_verb = 11:17;
% transit_noun = 39:45;

transit_dict = {'cheese_trans', 'coffee_trans', 'cofhoney_trans', 'hotdog_trans',...
    'peal_trans', 'peanut_trans', 'tea_trans'};
label_catg = {'Cheese','Coffee','CofHoney','Hotdog','Pealate','Peanut','Tea'};

% label_dict = cell(1,79);
% 
% label_verb = cell(1,1);
% id_verb = 1;
% label_noun = cell(1,1);
% id_noun = 1;

label_cnt = zeros(1,79);
label_cnt_time = zeros(1,79);
lab_dict_idx = 1;

% for i = 1:length(transit_dict)
%     label_dict{72+i} = transit_dict{i};
% end
% 
% for i = 1:length(transit_dict)
%     label_verb{10+i} = transit_dict{i};
% end
% 
% for i = 1:length(transit_dict)
%     label_noun{38+i} = transit_dict{i};
% end

dest_img_dir = './GTEA_dt';
dest_lab_dir = './GTEA_lab';
dest_verb_dir = './GTEA_verb';
dest_noun_dir = './GTEA_noun';

lab_dir = './gtea_labels_71/labels';
img_dir = './gtea_png/png';
activity_folder = './activity';

dir_lab = dir(lab_dir);
dir_lab(1:2) = [];
len_dir_lab = length(dir_lab);

dir_img = dir(img_dir);
dir_img(1:2) = [];
len_dir_img = length(dir_img);

%%

% lab_len1 = 72;
% len_s = zeros(1,1);
% 
% for i = 1:lab_len1
%     fd_name = label_dict{i};
%     split_name = split(fd_name,',');
%     len_s(i) = length(split_name);
%     verb_tmp = split_name{1};
%     if len_s(i) == 2
%         noun_tmp = split_name{2};
%     elseif len_s(i) == 3
%         noun_tmp = sprintf('%s,%s',split_name{2},split_name{3});
%     else
%         noun_tmp = sprintf('%s,%s,%s',split_name{2},split_name{3},split_name{4});
%     end
%     verb_id = find(strcmp(label_verb, verb_tmp));
%     noun_id = find(strcmp(label_noun, noun_tmp));
% 
%     mkfd_path = sprintf('%s/%s_%s', activity_folder, num2str(verb_id), num2str(noun_id));
%     mkdir(mkfd_path)
% end


%%
for i = 1:len_dir_lab
    i
    % disp(dir_lab(i).name);
    lab_name = dir_lab(i).name;
    lab_name = split(lab_name,'.');
    lab_name = lab_name{1};  
    
    lab_fd_name = split(lab_name,'_');
    lab_fd_name = lab_fd_name{2};
    idx_lab_fc = find(strcmp(label_catg, lab_fd_name));
    
    idx_lab = find(strcmp(label_catg, lab_fd_name));

    lab_path = sprintf('%s/%s', dir_lab(i).folder, dir_lab(i).name);
    img_path = sprintf('%s/%s',dir_img(i).folder, dir_img(i).name);
    img_dir_sub = dir(img_path);
    img_dir_sub(1:2) = [];
    
    fid = fopen(lab_path);
    cell_data = textscan(fid,'%s %s %s');
    idx_missing = find(ismissing(cell_data{1,3})==1);
    
    if ~isempty(idx_missing)
        lab_len = idx_missing(1)-1;
    else
        disp('no missing')
        lab_len = length(cell_data{1,3});
    end
    
    split_lab_time_end = strsplit(cell_data{2}{lab_len}, {'(',')','-'});
    lab_sub_end = str2num(split_lab_time_end{3});
    state = ones(1,lab_sub_end);
    state_verb = ones(1,lab_sub_end);
    state_noun = ones(1,lab_sub_end);
    
    for j = 1:lab_len
        lab_act = cell_data{1}{j};
        lab_time = cell_data{2}{j};
        
        split_lab_act = strsplit(lab_act,{'<','>'});
        split_lab_time = strsplit(lab_time, {'(',')','-'});
        
        lab_verb = split_lab_act{2};
        lab_noun = split_lab_act{3};
        lab_title = sprintf('%s,%s', lab_verb, lab_noun);
        
        lab_time_start = str2num(split_lab_time{2});
        lab_time_end = str2num(split_lab_time{3});

        label_ind = find(strcmp(label_dict, lab_title));
        label_verb_ind = find(strcmp(label_verb, lab_verb));
        label_noun_ind = find(strcmp(label_noun, lab_noun));

        state(lab_time_start:lab_time_end) = label_ind;
        state_verb(lab_time_start:lab_time_end) = label_verb_ind;
        state_noun(lab_time_start:lab_time_end) = label_noun_ind;

        idx_lab_cnt = find(strcmp(label_dict, lab_title));

        label_cnt(idx_lab_cnt) = label_cnt(idx_lab_cnt) + 1;
        label_cnt_time(idx_lab_cnt) = label_cnt_time(idx_lab_cnt) + (lab_time_end - lab_time_start);
        
        verb_id = find(strcmp(label_verb, lab_verb));
        noun_id = find(strcmp(label_noun, lab_noun));
        
        im_dir_path = sprintf('%s/%s_%s/%s_part_%s',activity_folder, num2str(verb_id), num2str(noun_id), lab_name, num2str(j) );
        mkdir(im_dir_path);
        range_img = lab_time_start:lab_time_end;
        
        for k = 1:length(range_img)
            im_order = k;
            im_path_store = sprintf('%s/%s',im_dir_path, img_dir_sub(range_img(k)).name);
            im_path_org = sprintf('%s/%s', img_dir_sub(range_img(k)).folder, img_dir_sub(range_img(k)).name);
            im = imread(im_path_org);
            im = imresize(im, [200,320]);
            imwrite(im, im_path_store);
        end
        
        if length(range_img) < 16
            disp(im_dir_path)
        end
        
    end
end

