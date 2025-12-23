# python
"""
stage1_to_hdf5.py

通用转换脚本说明（简短）：
- 扫描 source_dir 下的子目录，每个子目录视为一个 episode（或扫描 .npz 文件）。
- 优先读取文件名约定： images.npy | images.npz | qpos.npy | actions.npy | is_pad.npy
  或者 image_0000.png ... 的图片序列（支持多相机：image_cam0_0000.png / image_cam1_0000.png）。
- 将每个 episode 写入输出 HDF5 的 group: /episodes/{episode_name}/...
- 输出键： images (T, C, H, W) 或 (T, cam, C, H, W)； qpos (T, qdim)； actions (T, adim)； is_pad (T,)
- 可按需修改键名映射与读取逻辑以匹配实际数据格式。
"""

import os
import sys
import argparse
import numpy as np
import h5py
from PIL import Image
from glob import glob

def load_npy_or_npz(path):
    if path.endswith('.npz'):
        data = dict(np.load(path, allow_pickle=True))
        return data
    else:
        return np.load(path, allow_pickle=True)

def load_images_from_dir(img_files):
    # img_files: sorted list of file paths for one camera
    imgs = []
    for p in img_files:
        im = Image.open(p).convert('RGB')
        arr = np.array(im)  # H,W,C
        arr = np.transpose(arr, (2,0,1))  # C,H,W
        imgs.append(arr)
    return np.stack(imgs, axis=0)  # T,C,H,W

def gather_episode_data(ep_path, config):
    # config可指定文件名模式
    data = {}
    # try common npy/npz files
    for key in ['qpos','actions','is_pad','dones','rewards']:
        for ext in ('.npy', '.npz'):
            p = os.path.join(ep_path, key + ext)
            if os.path.exists(p):
                loaded = load_npy_or_npz(p)
                # if npz, prefer array with same name or first array
                if isinstance(loaded, dict):
                    # try key itself else first item
                    if key in loaded:
                        data[key] = loaded[key]
                    else:
                        first = next(iter(loaded.values()))
                        data[key] = first
                else:
                    data[key] = loaded
                break

    # images: try images.npy or image files
    imgs_path = os.path.join(ep_path, 'images.npy')
    if os.path.exists(imgs_path):
        imgs = np.load(imgs_path)
        # expected T,C,H,W or T,cam,C,H,W
        data['images'] = imgs
    else:
        # try patterns like image_cam{idx}_*.png or image_*.png
        files = sorted(glob(os.path.join(ep_path, 'image_*.png')))
        if len(files) > 0:
            # single camera
            data['images'] = load_images_from_dir(files)  # T,C,H,W
        else:
            # multi-camera pattern image_cam{c}_0000.png
            cam_files = {}
            for p in sorted(glob(os.path.join(ep_path, 'image_cam*_*.png'))):
                base = os.path.basename(p)
                # name like image_cam0_0000.png
                parts = base.split('_')
                if len(parts) >= 3:
                    cam = parts[1]  # cam0
                    cam_files.setdefault(cam, []).append(p)
            if cam_files:
                cams_sorted = sorted(cam_files.keys())
                cam_imgs = [load_images_from_dir(sorted(cam_files[c])) for c in cams_sorted]
                # stack to shape (T, num_cam, C, H, W)
                data['images'] = np.stack(cam_imgs, axis=1)
    return data

def write_episode_to_h5(h5f, ep_name, data, attrs):
    grp = h5f.create_group(f'episodes/{ep_name}')
    # write datasets if present
    for k, v in data.items():
        if v is None:
            continue
        # convert bool masks to uint8 etc.
        arr = np.array(v)
        # ensure consistent dtypes
        if arr.dtype == np.bool_:
            arr = arr.astype(np.uint8)
        grp.create_dataset(k, data=arr, compression='gzip')
    # write attrs
    for ak,av in attrs.items():
        grp.attrs[ak] = av

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True, help='source root, e.g. stage_1/FR_Gym')
    parser.add_argument('--out', required=True, help='output hdf5 file, e.g. out.hdf5')
    args = parser.parse_args()

    src = args.src
    out = args.out

    # collect episode dirs or npz files at top level
    candidates = []
    for p in sorted(os.listdir(src)):
        full = os.path.join(src, p)
        if os.path.isdir(full):
            candidates.append(full)
        elif p.endswith('.npz') or p.endswith('.npy'):
            # treat single-file episode
            candidates.append(full)

    if len(candidates) == 0:
        print('No episodes found in', src)
        sys.exit(1)

    # global attrs guessed from first episode
    global_attrs = {}
    with h5py.File(out, 'w') as h5f:
        for c in candidates:
            if c.endswith('.npz') or c.endswith('.npy'):
                # load single-file npz as an episode
                base = os.path.basename(c)
                ep_name = os.path.splitext(base)[0]
                loaded = load_npy_or_npz(c)
                if isinstance(loaded, dict):
                    data = {}
                    for k in ['images','qpos','actions','is_pad','dones','rewards']:
                        if k in loaded:
                            data[k] = loaded[k]
                else:
                    # fallback: store under data
                    data = {'data': loaded}
            else:
                ep_name = os.path.basename(c)
                data = gather_episode_data(c, {})
            # infer some attrs
            attrs = {}
            if 'actions' in data:
                attrs['action_dim'] = int(np.array(data['actions']).shape[-1])
            if 'qpos' in data:
                attrs['qpos_dim'] = int(np.array(data['qpos']).shape[-1])
            # write
            write_episode_to_h5(h5f, ep_name, data, attrs)
            print('wrote', ep_name)
        # root attrs
        h5f.attrs['source'] = src
        print('done. output:', out)

if __name__ == '__main__':
    main()