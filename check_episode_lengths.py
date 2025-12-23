import h5py
import os
import argparse
import fnmatch

def check_episode_lengths(dataset_dir):
    """
    Scans a directory for HDF5 files and prints the length of the action trajectory for each file.

    Args:
        dataset_dir (str): The path to the directory containing the HDF5 files.
    """
    print(f"Scanning directory: {dataset_dir}")

    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            hdf5_files.append(os.path.join(root, filename))

    if not hdf5_files:
        print(f"No HDF5 files found in '{dataset_dir}'.")
        return

    print(f"Found {len(hdf5_files)} hdf5 files.")
    print("\n--- Episode Lengths (Action Timesteps) ---")

    all_lengths = []
    for hdf5_file in sorted(hdf5_files):
        try:
            with h5py.File(hdf5_file, 'r') as root:
                if '/action' in root:
                    action_len = root['/action'].shape[0]
                    all_lengths.append(action_len)
                    print(f"  - File: {os.path.basename(hdf5_file)}, Length: {action_len}")
                else:
                    print(f"  - File: {os.path.basename(hdf5_file)}, WARNING: '/action' dataset not found.")
        except Exception as e:
            print(f"  - File: {os.path.basename(hdf5_file)}, ERROR: Could not read file. Reason: {e}")

    print("------------------------------------------\n")

    if all_lengths:
        print("Summary:")
        print(f"  - Min length: {min(all_lengths)}")
        print(f"  - Max length: {max(all_lengths)}")
        print(f"  - Avg length: {sum(all_lengths) / len(all_lengths):.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Check the length of action trajectories in HDF5 dataset files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        help="Path to the dataset directory to scan.",
        required=True
    )
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(f"Error: Directory not found at '{args.dataset_dir}'")
    else:
        check_episode_lengths(args.dataset_dir)

