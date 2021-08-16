clc;clear

src_dir = './activity';
dest_dir = './Activity_split';

dir_src = dir(src_dir);
dir_src(1:2) = [];
len_dir_src = length(dir_src);

for i = 1:len_dir_src
    dir_name = dir_src(i).name;
    
    dir_path = sprintf('%s/%s', dir_src(i).folder, dir_src(i).name);
    dir_sub = dir(dir_path);
    dir_sub(1:2) = [];
    
    len_dir_sub = length(dir_sub);
    
    for j = 1:len_dir_sub
        sub_name = dir_sub(j).name;
        s_name_split = split(sub_name,'_');
        prex_name = s_name_split{1};
        
        src_fd = sprintf('%s/%s', dir_sub(j).folder, dir_sub(j).name);
        
        if prex_name == 'S4'		% select
            disp('s4')
            activity_fd = sprintf('%s/%s/%s',dest_dir,'test', dir_name);
            mkdir(activity_fd);
            dest_fd = sprintf('%s/%s', activity_fd, sub_name);
            % copy folder 
            copyfile(src_fd, dest_fd)
        else
            activity_fd = sprintf('%s/%s/%s',dest_dir, 'train', dir_name);
            mkdir(activity_fd);
            dest_fd = sprintf('%s/%s', activity_fd, sub_name);
            % copy folder 
            copyfile(src_fd, dest_fd)
        end
           
    end
    
    
end
