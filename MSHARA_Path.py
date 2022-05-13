class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'face':
            root_dir = './data/face'                # root_dir and seq_lab_dir need to change according to local path
            flow_dir = './data/flow_img'            # optical flow images are not used in this stage

            # Save preprocess data into output_dir  # output_dir can be ignored as well
            output_dir = './data/face'
            seq_lab_dir = './data/clip_lab'

        elif database == 'gtea':
            root_dir = 'F:\Crome\GTEA/GTEA_dt'
            flow_dir = 'F:\Crome\GTEA/GTEA_dt'
            # Save preprocess data into output_dir
            output_dir = 'F:\Crome\GTEA/GTEA_dt'
            seq_lab_dir = 'F:\Crome\GTEA/GTEA_lab'

        return root_dir, output_dir, seq_lab_dir, flow_dir


