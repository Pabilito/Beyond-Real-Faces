# Synthetic facial recognition datasets evaluation replication package

The list of packages installed in the conda environment used in the study can be found in *environment.yml*

Replace subsequent appropriate placeholder parameter values (IN UPPERCASE) with actual values.

Store your dataset in ./Dataset. Due to size constraints, we do not provide the datasets and the pretrained weights (to be stored as ./Models/ArcFace_R100_MS1MV3.pth). The model can be found here https://github.com/deepinsight/insightface/tree/master/model_zoo. If you decide to choose a different model, adjust the *Embedding_Computation/analyze_embeddings.py* accordingly.

## Embedding_Computation - Compute embeddings for a dataset
```bash
python compute_embeddings_to_json.py
```
*Output*: JSON file with embeddings.

## Mated_vs_non-mated - Compare mated and non-mated distributions of a dataset.

Use previously generated JSON files to compute distributions:
```bash
python analyze_embeddings.py INPUT.json --n_comparisons 10000
```
*Output*:
![Vec2Face_MatedVsNonmated_1000000](https://github.com/user-attachments/assets/9eb631f9-2528-4a47-b8e8-a7a636c5041d)

If you want to get more detailed metrics use:
```bash
python compute_similarities.py INPUT.json --n_comparisons 10000
python analyze_embeddings_with_metrics.py
```
*Output*:
![CemiFace_MatedVsNonmated_2000000](https://github.com/user-attachments/assets/67b19f7d-db34-4458-a9ca-e0a8b8e1850f)

## CASIA_compare - Check how closely related a dataset is with respect to CASIA-WebFace.
Find the best matches in CASIA and plot the similarity distribution based on two JSON input files.
```bash
python compare_embeddings.py INPUT.json --n_comparisons 10000 --casia_file CASIA.json
```
*Output*:
![Vec2Face_BestMatches_10000](https://github.com/user-attachments/assets/4f6ca291-473e-4e13-8421-c81a3fce2941)

Alternatively, just save the values of closest samples:
```bash
python compute_similarities.py INPUT.json --n_comparisons 10000 --casia_file CASIA.json
```
*Output*: SimilarityScores/INPUT_BestMatches_10000.txt

Alternatively, find the closest samples with respect to the CASIA datasets, use:
```bash
./find_closest.sh
```
*Output*: JSON file with the list of the closest samples.

Then save the paths in a simplified CSV file:
```bash
./get_paths.sh 
```
*Output*: SIMILAR_IMAGES.csv

To plot the results, use:
(.tar.gz should also work)
```bash
python plot_similar.py SIMILAR_IMAGES.csv --casia-file CASIA.zip --other-file OTHER.zip --samples 5
```
*Output*:
![comparison_IDiffFace_5_samples](https://github.com/user-attachments/assets/24315218-816c-4c1c-a506-804a9775152f)

