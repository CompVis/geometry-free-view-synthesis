import argparse, os, sys, glob, shutil, subprocess
import numpy as np


def pose_to_sparse(txt_src, img_src, spa_dst, DEBUG=False, exists_ok=True,
                   worker_idx=None, world_size=None):
    opt = argparse.Namespace(txt_src=txt_src, img_src=img_src, spa_dst=spa_dst)

    assert os.path.exists(opt.txt_src), opt.txt_src
    assert os.path.exists(opt.img_src), opt.img_src

    if not os.path.exists(opt.spa_dst):
        os.makedirs(opt.spa_dst)
    else:
        if DEBUG:
            shutil.rmtree(opt.spa_dst)
        elif not exists_ok:
            print("Output directory exists, doing nothing")
            return 0


    txts = sorted(glob.glob(os.path.join(opt.txt_src, "*.txt")))
    if DEBUG: print(txts)

    if worker_idx is not None and world_size is not None:
        txts = txts[worker_idx::world_size]

    if DEBUG: txts = txts[:1]
    failed = list()
    for txt in txts:
        vidid = os.path.splitext(os.path.split(txt)[1])[0]
        print(f"Processing {vidid}")

        if (os.path.exists(os.path.join(opt.spa_dst, vidid, "sparse", "cameras.bin")) and
                os.path.exists(os.path.join(opt.spa_dst, vidid, "sparse", "images.bin")) and
                os.path.exists(os.path.join(opt.spa_dst, vidid, "sparse", "points3D.bin"))):
            print("Found sparse model, skipping {}".format(vidid))
            continue

        if os.path.exists(os.path.join(opt.spa_dst, vidid)):
            shutil.rmtree(os.path.join(opt.spa_dst, vidid))
            print("Found partial output of previous run, removed {}".format(vidid))


        # read camera poses for this sequence
        with open(txt, "r") as f:
            firstline = f.readline()

        if firstline.startswith("http"):
            if DEBUG: print("Ignoring first line.")
            skiprows = 1
        else:
            skiprows = 0

        vid_data = np.loadtxt(txt, skiprows=skiprows)
        if len(vid_data.shape) != 2:
            failed.append(vidid)
            print(f"Wrong txt format for {vidid}!")
            continue

        timestamps = vid_data[:,0].astype(np.int)
        if DEBUG: print(timestamps)
        filenames = [str(ts)+".png" for ts in timestamps]

        if not len(filenames) > 1:
            failed.append(vidid)
            print(f"Less than two frames, skipping {vidid}!")
            continue

        if not os.path.exists(os.path.join(opt.img_src, vidid)):
            failed.append(vidid)
            print(f"Could not find frames, skipping {vidid}!")
            continue

        if not len(glob.glob(os.path.join(opt.img_src, vidid, "*.png"))) == len(filenames):
            failed.append(vidid)
            print(f"Could not find all frames, skipping {vidid}!")
            continue

        if DEBUG: print(vid_data[0,1:])
        K_params = vid_data[:,1:7]
        Ks = np.zeros((K_params.shape[0], 3, 3))
        Ks[:,0,0] = K_params[:,0]
        Ks[:,1,1] = K_params[:,1]
        Ks[:,0,2] = K_params[:,2]
        Ks[:,1,2] = K_params[:,3]
        Ks[:,2,2] = 1
        assert (Ks[0,...]==Ks[1,...]).all()
        K = Ks[0]
        if DEBUG: print(K)

        Rts = vid_data[:,7:].reshape(-1, 3, 4)
        if DEBUG: print(Rts[0])

        # given these intrinsics and extrinsics, find a sparse set of scale
        # consistent 3d points following
        # https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses

        # extract and match features on frames
        dst_dir = os.path.join(opt.spa_dst, vidid)
        os.makedirs(dst_dir)
        database_path = os.path.join(dst_dir, "database.db")

        # symlink images
        image_path = os.path.join(dst_dir, "images")
        os.symlink(os.path.abspath(os.path.join(opt.img_src, vidid)), image_path)

        cmd = ["colmap", "feature_extractor",
               "--database_path", database_path,
               "--image_path", image_path,
               "--ImageReader.camera_model", "PINHOLE",
               "--ImageReader.single_camera", "1",
               "--SiftExtraction.use_gpu", "1"]
        if DEBUG: print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        # read the database
        from database import COLMAPDatabase, blob_to_array, array_to_blob
        db = COLMAPDatabase.connect(database_path)

        # read and update camera
        ## https://colmap.github.io/cameras.html
        cam = db.execute("SELECT * FROM cameras").fetchone()
        camera_id = cam[0]
        camera_model = cam[1]
        assert camera_model == 1 # PINHOLE
        width = cam[2]
        height = cam[3]
        params = blob_to_array(cam[4], dtype=np.float64)
        assert len(params) == 4 # fx, fy, cx, cy for PINHOLE

        # adjust params
        params[0] = width*K[0,0]
        params[1] = height*K[1,1]
        params[2] = width*K[0,2]
        params[3] = height*K[1,2]
        
        # update
        db.execute("UPDATE cameras SET params = ?  WHERE camera_id = ?",
                   (array_to_blob(params), camera_id))
        db.commit()

        # match features
        cmd = ["colmap", "sequential_matcher",
               "--database_path", database_path,
               "--SiftMatching.use_gpu", "1"]
        if DEBUG: print(" ".join(cmd))
        subprocess.run(cmd, check=True)

        # triangulate
        ## prepare pose model
        ### https://colmap.github.io/format.html#text-format
        pose_dir = os.path.join(dst_dir, "pose")
        os.makedirs(pose_dir)
        cameras_txt = os.path.join(pose_dir, "cameras.txt")
        with open(cameras_txt, "w") as f:
            f.write("{} PINHOLE {} {} {}".format(camera_id, width, height,
                                                 " ".join(["{:.2f}".format(p) for p in params])))

        images_txt = os.path.join(pose_dir, "images.txt")
        # match image ids with filenames and export their extrinsics to images.txt
        images = db.execute("SELECT image_id, name, camera_id FROM images").fetchall()
        lines = list()
        for image in images:
            assert image[2] == camera_id
            image_id = image[0]
            image_name = image[1]
            image_idx = filenames.index(image_name)
            Rt = Rts[image_idx]
            R = Rt[:3,:3]
            t = Rt[:3,3]
            # convert R to quaternion
            from scipy.spatial.transform import Rotation
            Q = Rotation.from_matrix(R).as_quat()
            # from x,y,z,w to w,x,y,z
            line = " ".join(["{:.6f}".format(x) for x in [Q[3],Q[0],Q[1],Q[2],t[0],t[1],t[2]]])
            line = "{} ".format(image_id)+line+" {} {}".format(camera_id, image_name)
            lines.append(line)
            lines.append("") # empty line for 3d points to be triangulated
        with open(images_txt, "w") as f:
            f.write("\n".join(lines)+"\n")

        # create empty points3D.txt
        points3D_txt = os.path.join(pose_dir, "points3D.txt")
        open(points3D_txt, "w").close()

        # run point_triangulator
        out_dir = os.path.join(dst_dir, "sparse")
        os.makedirs(out_dir)
        cmd = ["colmap", "point_triangulator",
               "--database_path", database_path,
               "--image_path", image_path,
               "--input_path", pose_dir,
               "--output_path", out_dir]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Triangulation failed for {vidid}!")
            failed.append(vidid)

    print("Failed sequences:")
    print("\n".join(failed))
    print(f"Could not create sparse models for {len(failed)} sequences.")
    return len(txts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--txt_src', type=str,
                        help='path to directory containing <vidid>.txt files of realestate format')
    parser.add_argument('--img_src', type=str,
                        help='path to directory containing <vidid>/<timestamp>.png frames')
    parser.add_argument('--spa_dst', type=str,
                        help='path to directory to write sparse models into')
    parser.add_argument('--DEBUG', action="store_true",
                        help='for quick development')
    parser.add_argument('--worker_idx', type=int,
                        help='if world_size is specified, should be 0<=worker_idx<world_size, otherwise ignored')
    parser.add_argument('--world_size', type=int,
                        help='how many workers there will be overall')

    opt = parser.parse_args()
    if opt.world_size is not None:
        assert opt.world_size > 1
        assert opt.worker_idx is not None
        assert 0<=opt.worker_idx<opt.world_size

    n_sparsified = pose_to_sparse(opt.txt_src, opt.img_src, opt.spa_dst,
                                  DEBUG=opt.DEBUG,
                                  worker_idx=opt.worker_idx,
                                  world_size=opt.world_size)

    print(f"Sparsified {n_sparsified} sequences.")
