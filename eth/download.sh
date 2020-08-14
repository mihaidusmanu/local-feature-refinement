# Create data directory.
mkdir ETH3D

# cd into data directory.
cd ETH3D

# Download undistorted images.
wget https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z
7z x multi_view_training_dslr_undistorted.7z
rm multi_view_training_dslr_undistorted.7z

# Download ground-truth scans.
wget https://www.eth3d.net/data/multi_view_training_dslr_scan_eval.7z
7z x multi_view_training_dslr_scan_eval.7z
rm multi_view_training_dslr_scan_eval.7z

# Download the match-lists.
wget https://dsmn.ml/files/local-feature-refinement/ETH3D-match-lists.tar
tar xvf ETH3D-match-lists.tar
rm ETH3D-match-lists.tar

# cd out of data directory.
cd ..
