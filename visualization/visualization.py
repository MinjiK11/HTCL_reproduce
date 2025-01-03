import numpy as np
import open3d as o3d
import os
from laserscan import LaserScan, SemLaserScan
import yaml
import glob

def get_ref_3d(voxel_size):
    scene_size = (51.2, 51.2, 6.4)
    vox_origin = np.array([0, -25.6, -2])
    voxel_size = 0.2

    vol_bnds = np.zeros((3,2))
    vol_bnds[:,0] = vox_origin
    vol_bnds[:,1] = vox_origin + np.array(scene_size)

    # Compute the voxels index in lidar cooridnates
    vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
    idx = np.array([range(vol_dim[0]*vol_dim[1]*vol_dim[2])])
    xv, yv, zv = np.meshgrid(range(vol_dim[0]), range(vol_dim[1]), range(vol_dim[2]), indexing='ij')
    vox_coords = np.concatenate([xv.reshape(1,-1), yv.reshape(1,-1), zv.reshape(1,-1)], axis=0).astype(int).T
        
    return vox_coords

def learning_map_inv(y_pred,filename):
    """Save predictions for evaluations and visualizations.

    learning_map_inv: inverse of previous map
        
    0: 0    # "unlabeled/ignored"  # 1: 10   # "car"        # 2: 11   # "bicycle"       # 3: 15   # "motorcycle"     # 4: 18   # "truck" 
    5: 20   # "other-vehicle"      # 6: 30   # "person"     # 7: 31   # "bicyclist"     # 8: 32   # "motorcyclist"   # 9: 40   # "road"   
    10: 44  # "parking"            # 11: 48  # "sidewalk"   # 12: 49  # "other-ground"  # 13: 50  # "building"       # 14: 51  # "fence"          
    15: 70  # "vegetation"         # 16: 71  # "trunk"      # 17: 72  # "terrain"       # 18: 80  # "pole"           # 19: 81  # "traffic-sign"
    """

    y_pred[y_pred==10] = 44
    y_pred[y_pred==11] = 48
    y_pred[y_pred==12] = 49
    y_pred[y_pred==13] = 50
    y_pred[y_pred==14] = 51
    y_pred[y_pred==15] = 70
    y_pred[y_pred==16] = 71
    y_pred[y_pred==17] = 72
    y_pred[y_pred==18] = 80
    y_pred[y_pred==19] = 81
    y_pred[y_pred==1] = 10
    y_pred[y_pred==2] = 11
    y_pred[y_pred==3] = 15
    y_pred[y_pred==4] = 18
    y_pred[y_pred==5] = 20
    y_pred[y_pred==6] = 30
    y_pred[y_pred==7] = 31
    y_pred[y_pred==8] = 32
    y_pred[y_pred==9] = 40


    y_pred_bin = y_pred.astype(np.uint16)
    y_pred_bin.tofile(os.path.join('HTCL/code/data/GT',filename[:-4]+'.label'))

def npy2bin(path):
    for f in glob.glob(os.path.join(path,'*_1_1.npy')):
        filename=os.path.basename(f)
        file=np.load(f)
        learning_map_inv(file,filename)

data_root='HTCL/code/data/GT'
# label_tag='predictions'
# label_path=os.path.join(data_root,label_tag)
label_path=data_root
point_folder=os.path.join(data_root,'points')
vis_folder=os.path.join(data_root,'visualization')

file=open('HTCL/code/tools/data_converter/kitti_process/semantic-kitti.yaml')
info=yaml.full_load(file)
color_dict=info['color_map'] # label별 색깔에 대한 정보

vis=o3d.visualization.Visualizer()
vis.create_window()

npy2bin("HTCL/code/data/occupancy/semanticKITTI/lss-semantic_kitti_voxel_label/labels/08")

for f in glob.glob(os.path.join(label_path,'*.label')):
    print(f)
    label=np.fromfile(f,dtype=np.uint16)[None]

    # filename=os.path.basename(f)
    # learning_map_inv(label,filename)

    vox_coords=get_ref_3d(0.2)
    unmasked_idx=np.asarray(np.where((label.reshape(-1)>0)&(label.reshape(-1)!=255))).astype(np.int32)

    vox_coords=vox_coords[unmasked_idx[0],:]
    vox_coords=np.array(vox_coords,dtype=np.str_)

    if not os.path.exists(point_folder):
        os.makedirs(point_folder)

    filename = os.path.basename(f)
    frame_id = os.path.splitext(filename)[0]
    txt_file= os.path.join(point_folder,frame_id+'.txt') 

    with open(txt_file,"w") as file:
        for line in vox_coords:
            file.write(" ".join(line)+"\n")

    bin_file=os.path.join(point_folder,frame_id+'.bin')
    pts_bin = vox_coords.astype(np.uint8)
    pts_bin.tofile(bin_file)


    scan=LaserScan() 
    semScan=SemLaserScan(color_dict)

    scan.open_scan(bin_file)
    semScan.points=scan.points

    semScan.open_label(f)

    # print(semScan.sem_label)
    # print(semScan.inst_label)
    semScan.colorize()
    # print(semScan.sem_label_color)
    # print(semScan.inst_label_color)

    xyz=np.loadtxt(txt_file)
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(xyz)
    pcd.colors=o3d.utility.Vector3dVector(semScan.sem_label_color)
    vis.add_geometry(pcd)
    #vis.visualization.draw_geometries([pcd])
    ctr = vis.get_view_control()
    # ctr.set_front([-0.5, 0.5, 0.3])
    ctr.set_front([-0.7044, 0.0616, 0.3])
    ctr.set_up([0, 0, 1])

    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)
    vis_file=os.path.join(vis_folder,frame_id+'.png')
    vis.capture_screen_image(vis_file,True)
    vis.remove_geometry(pcd)