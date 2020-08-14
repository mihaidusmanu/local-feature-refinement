if [ $# -ne 1 ]; then
    echo "Usage: bash eth/prepare_dataset.sh path_to_colmap_executable_folder"
    exit
fi

COLMAP_PATH=$1

for dir in `ls ETH3D`; do
    python utils/create_starting_database_eth.py --colmap_path $COLMAP_PATH --dataset_path ETH3D/$dir
    python utils/create_image_list_file.py --dataset_path ETH3D/$dir
done;