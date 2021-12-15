# Multi-View Optimization of Local Feature Geometry

This repository contains the implementation of the following paper:

```text
"Multi-View Optimization of Local Feature Geometry".
M. Dusmanu, J.L. Schönberger, and M. Pollefeys. European Conference on Computer Vision 2020 (Oral).
```

[[Paper on arXiv]](https://arxiv.org/abs/2003.08348) [[Project page]](https://dsmn.ml/publications/mvolfg.html) [[Qualitative results]](https://youtu.be/eH4UNwXLsyk)
    
## Requirements

### C++

[COLMAP](https://colmap.github.io/) should be installed as a library before proceeding. Please refer to the official website for installation instructions. For the paper, we have used the `dev` branch of COLMAP at commit `f4eaade` (you can run `git checkout f4eaade` before compiling COLMAP to use the same version). **We recommend setting an environmental variable to the colmap folder by running `COLMAP_PATH=path_to_colmap_executable_folder`.**

The only additional requirement is `protobuf` which can be installed on Ubuntu as follows `sudo apt install protobuf-compiler libprotobuf-dev`.

Start by parsing the `proto` file and generate output for both Python and C++:
```bash
protoc --python_out=two-view-refinement/ --python_out=reconstruction-scripts/ --cpp_out=multi-view-refinement/ types.proto
```

Now, the multi-view refinement code can be compiled as follows:
```bash
cd multi-view-refinement
mkdir build; cd build
cmake ..
make
```

### Python

Python 3.6+ is recommended for running our code. [Conda](https://docs.conda.io/en/latest/) can be used to create a new environment and install the required packages:
```bash
conda create -n local-feature-refinement -y
conda activate local-feature-refinement

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch -y
conda install opencv imagesize tqdm protobuf -c conda-forge -y
```

## Extracting features

<details>
<summary>Click for details...</summary>

In order to make our evaluation reproducible regardless of updates to the repositories of individual features, we have forked all repositories at the point in time when we evaluated them. Please refer to the individual repositories for installation instructions.

### [SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)

We used the GPU-SIFT distribution coming with COLMAP. You can use the following command to extract features:
```bash
python utils/extract_features_sift.py --colmap_path $COLMAP_PATH --image_path path_to_images
```

### [SURF](http://people.ee.ethz.ch/~surf/eccv06.pdf)

We used the OpenCV implementation. You can use the following command to extract features:
```bash
python utils/extract_features_surf.py --image_path path_to_images
```

### [D2-Net](https://arxiv.org/abs/1905.03561)

Clone the repository (`git clone git@github.com:mihaidusmanu/d2-net.git; git checkout 2a4d88f`) and use the following command to extract features:
```bash
python extract_features.py --image_list_file image_list.txt (--multiscale)
```

### [Key.Net](https://arxiv.org/abs/1904.00889)

Clone the fork (`git clone git@github.com:mihaidusmanu/Key.Net.git; git checkout local-feature-refinement`) and use the following command to extract features:
```bash
python extract_multiscale_features.py --list_images image_list.txt
```

### [R2D2](https://arxiv.org/abs/1906.06195)

Clone the fork (`git clone git@github.com:mihaidusmanu/r2d2.git`) and use the following command to extract features:
```bash
python extract.py --images image_list.txt
```

### [SuperPoint](https://arxiv.org/abs/1712.07629)

Clone the fork (`git clone git@github.com:mihaidusmanu/SuperPointPretrainedNetwork.git`) and use the following command to extract features:
```bash
python extract_features_superpoint_list.py image_list.txt
```

### Image lists

To create the image lists, you can use the provided utility `utils/create_image_list_file.py`.

</details>

## Running the Local Feature Evaluation Benchmark

<details>
<summary>Click for details...</summary>

Once the multi-view refinement code was compiled successfully, the environment was created, and you made sure that you can run feature extraction, you can try out the Local Feature Evaluation Benchmark. To make sure that everything is working properly, we recommend starting on the two small datasets (Fountain and Herzjesu). You can download the datasets by running `bash local-feature-evaluation/download.sh` (~4GB required).

The evaluation can be run using the following command:
```bash
python local-feature-evaluation/benchmark.py --colmap_path $COLMAP_PATH --dataset_name dataset_name --method_name method_name
```

For instance, in order to evaluate SIFT on Fountain, one would run:
```bash
python local-feature-evaluation/benchmark.py --colmap_path $COLMAP_PATH --dataset_name Fountain --method_name sift
```
This will produce two output files: `output/sift-Fountain-ref.txt` and `output/sift-Fountain-raw.txt` containing `json` objects with reconstruction statistics for features with and without refinement, respectively.

Similarly to the paper, `local-feature-evaluation/compare_reconstructions.py` can be used to compare a refined reconstruction and its raw counterpart on commonly registered images only.

</details>

## Running Two-View-Refinement Benchmark on HPatches Dataset

<details>
<summary>Click for details...</summary>

The download of HPatches dataset is included in the `local-feature-evaluation/download.sh`. It follows the d2-net instructions for filtering out the high resolution sequences. 

Since HPatches dataset has many sequences, we need to extract features for all subdirectories before running refinement. To do that, run

```bash
python utils/extract_features_all_subdirectories.py --directory_path LFE/hpatches-sequences-release --method_name method_name
```

For instance, in order to extract SURF features on all subdirectories of HPatches, run

```bash
python utils/extract_features_all_subdirectories.py --directory_path LFE/hpatches-sequences-release --method_name surf
```

After preparing the dataset and extracting the features, to run the two-view-refinement bench:

```bash
python two-view-refinement/eval_hpatches.py --dataset_name hpatches-sequences-release --method_name surf
```
As before, you can replace SURF with any other local supported in this repo as `--method_name`. If you want to skip the patch-flow refinement, you can add `--skip_refinement` flag.

The output will be an image with MMA evaluation, and also a `.npy` file containing the error statistics summary in the `cache` folder.

</details>

## Running the ETH3D Triangulation Benchmark

<details>
<summary>Click for details...</summary>

You can download and prepare the dataset by running `bash eth/download.sh; bash eth/prepare_dataset.sh $COLMAP_PATH` (~15GB required). Please follow the official instructions to install the [ETH3D multi-view evaluation program](https://github.com/ETH3D/multi-view-evaluation). **We recommend setting an environmental variable to the evaluation folder by running `EVAL_PATH=path_to_evaluation_executable_folder`.**

The evaluation can be run using the following command:
```bash
python eth/benchmark.py --colmap_path $COLMAP_PATH --evaluation_path $EVAL_PATH --dataset_name dataset_name --method_name method_name
```

For instance, in order to evaluate SIFT on pipes, one would run:
```bash
python eth/benchmark.py --colmap_path $COLMAP_PATH --evaluation_path $EVAL_PATH --dataset_name pipes --method_name sift
```
This will produce two output files: `output/sift-pipes-ref.txt` and `output/sift-pipes-raw.txt` containing sparse triangulation accuracy and completeness for features with and without refinement, respectively.

</details>

## Running in a custom scenario

<details>
<summary>Click for details...</summary>

### Custom dataset

In order to facilitate the use of our method with custom datasets, we provide several helpful scripts:
- `utils/create_starting_database.py` creates an initial database containing images and camera information from EXIF data.
- `utils/create_image_list_file.py` creates a list of images `image-list.txt` from a database.
- `utils/create_exhaustive_matching_file.py` creates an exhaustive list of image pairs to match `match-list.txt` from a database.

### Custom features

The proposed method works regardless of local features used. You can provide your own features in [`npz`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) files that encapsulate two arrays: 

- `keypoints` - `N x 2` - array containing the positions of keypoints `x, y`. The `X` axis is pointing to the right and the `Y` axis to the bottom.
- `descriptors` - `N x D` - array containing the L2 normalized descriptors.

### Reconstruction

We suppose the dataset directory has the following structure:
```
.
├── images
│  └── *.{jpg | png | ...}
│  └── *.{jpg | png | ...}.method_name (npz files with features)
├── database.db (created by utils/create_starting_database.py)
├── image-list.txt (created by utils/create_image_list_file.py)
└── match-list.txt (created by utils/create_exhaustive_matching_file.py for instance)
```

The list of image pairs to match `match-list.txt` can be replaced by a partial list. For image datasets extracted from videos, you can use sequential matching (i.e, last 10-20 frames). For large datasets (>100 images), we suggest using retrieval first and only matching with respect to the closest 20-50 images.

To run the refinement pipeline followed by 3D reconstruction with both refined and raw features, you can use:
```bash
python custom_demo.py --colmap_path $COLMAP_PATH --dataset_name dataset_name --dataset_path path_to_dataset --method_name method_name
```
The output is the same as for the Local Feature Evaluation Benchmark. You can then use COLMAP to visualize the resulting reconstructions.

If you are using a method that's not part of our initial evaluation, don't forget to add the feature extraction resolution and matching parameters to `max_size_dict` and `matcher_dict` respectively at the top of `custom_demo.py`.

</details>

## Benchmarking new methods without refinement

The two-view refinement can be skipped completely by setting the following environmental variable `SKIP_REFINEMENT=1`. For instance, in order to evaluate SIFT without refinement on pipes, one would run:
```bash
SKIP_REFINEMENT=1 python eth/benchmark.py --colmap_path $COLMAP_PATH --evaluation_path $EVAL_PATH --dataset_name pipes --method_name sift
```

## Coming soon

This repository will be updated during the following months with

- [x] Local Feature Evaluation benchmark code and instructions
- [ ] HPatches Sequences matching evaluation code and instructions
- [x] ETH3D triangulation evaluation code and instructions
- [ ] ETH3D localization evaluation code and instructions
- [ ] Training data and scripts

## BibTeX

If you use this code in your project, please cite the following paper:
```bibtex
@InProceedings{Dusmanu2020Multi,
    author = {Dusmanu, Mihai and Sch\"onberger, Johannes L. and Pollefeys, Marc},
    title = {{Multi-View Optimization of Local Feature Geometry}},
    booktitle = {Proceedings of the European Conference on Computer Vision},
    year = {2020},
}
```
