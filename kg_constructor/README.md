# Knowledge Graph (KG) Constructor
This folder contains source code responsible for 1) creating the KG using knowledge extracted from the datasets available online and 2) resolving any inconsistencies that may rise during the KG construction process.

## Directories
Following is a description of the directories.

* <code>[./configuration](./configuration)</code>: Contains **.ini** files used for setting up the configurations.
* <code>[./data](./data)</code>: Contains the dataset used to create the KG. It also contains other data files used as an input for creating the KG. For details about the files in this directory, please refer to its own [README](./data/README.md) file.
* <code>[./integrate_modules](./integrate_modules)</code>: Contains source code for creating the knowledge graph and doing inconsistency resolution.
* <code>[./output](./output)</code>: All output files will end up here.
* <code>[./postprocess_modules](./postprocess_modules)</code>: Contains source code for postprocess.
* <code>[./tools](./tools)</code>: Utility files.

## How to Run

### Step 1: Update the paths.
Modify the file paths in the following files to match your local settings.

* <code>[./data/data_path_file.txt](./data/data_path_file.txt)</code>
* <code>[./configuration/create_kg_config.ini](./configuration/create_kg_config.ini)</code>
* <code>[./configuration/postprocess_config.ini](./configuration/postprocess_config.ini)</code>

### Step 2: Clean the output directory.
Current <code>[output](./output)</code> directory consists of results used in the paper. If you wish to run the code and obtain new results, please remove all files and directories under it.

```
rm -r ./output/*
```

### Step 3: Run the KG construction and inconsistency resolver.
This step will construct the inconsistency-free knowledge graph. The generated files will be populated under the <code>[output](./output)</code> directory.

```
python3 create_kg.py --phase all
```

### Step 4: Run postprocessing.
Postprocess the knowledge graph created in Step 3 to generate Hypothesis Generator friendly files.

```
./run_postprocess.sh
```

## Output
Following files and directories will be populated under the <code>[./output](./output)</code> directory once finished running.

* <code>[./output/final](./output/final)</code>: Contains all the files to be used for training the final model.
* <code>[./output/folds](./output/folds)</code>: Contains all the files to be used for doing the k-fold cross validation.
* <code>[./output/data.txt](./output/data.txt)</code>: Same file as <code>[./output/kg_final.txt](./output/kg_final.txt)</code> with subset of columns *Subject*, *Predicate*, and *Object*. Also, the *Label* column is added.
* <code>[./output/entities.txt](./output/entities.txt)</code>: All entities of the knowledge graph.
* <code>[./output/entity_full_names.txt](./output/entity_full_names.txt)</code>: All entities of the knowledge graph with their corresponding entity types.
* <code>[./output/hypotheses.txt](./output/hypotheses.txt)</code>: Hypotheses that needs to be generated using the Hypothesis Generator.
* <code>[./output/kg_final.txt](./output/kg_final.txt)</code>: Final knowledge graph produced by the Knowledge Graph Constructor. This file includes the resolved inconsistencies. The Knowledge is represented in the triplet format. There are 6 columns in the following order: ***Subject***, ***Predicate***, ***Object***, ***Belief*** (The confidence score of this specific fact measured by the inconsistency corrector. Default is AverageLog), ***Source size*** (The number of sources supporting this knowledge), and ***Sources*** (The list of sources supporting this knowledge.)
* <code>[./output/kg_without_inconsistencies.txt](./output/kg_without_inconsistencies.txt)</code>: Same file as the [./output/kg_final.txt](./output/kg_final.txt) except for the absence of resolved inconsistencies.
* <code>[./output/relations.txt](./output/relations.txt)</code>: All the relations in the knowledge graph.
* <code>[./output/resolved_inconsistencies.txt](./output/resolved_inconsistencies.txt)</code>: File containing the inconsistencies resolved through the computational method. The first six columns are for the fact with the highest belief (i.e. one that the inconsistency corrector think it is correct) among conflicting facts. There columns are in the following order: ***Subject***, ***Predicate***, ***Object***,***Belief*** (The confidence score of this specific fact measured by the inconsistency corrector. Default is AverageLog.), ***Source size*** (The number of sources supporting this knowledge.), ***Sources*** (The list of sources supporting this knowledge.), ***Total source size*** (The number of all sources including the source size from the conflicting facts.), ***Mean belief of conflicting tuples*** (The average of beliefs of all other tuples conflicting to one that is represented in the first 3 columns.), ***Belief difference*** (The difference between the *Belief* and the *Mean belief of conflicting tuples*.), and ***Conflicting tuple info*** (The list of all the tuples conflicting to one that is represented in the first 3 columns. It is the list represented in "[(element1), (element2), ...]" and each element of the list is represented by "tuple, sources, belief" where tuple is "(subject, predicate, object)", and source is "[source1, source2, ...]".)
* <code>[./output/trustworthiness_data_summary.pdf](./output/trustworthiness_data_summary.pdf)</code>: Figure showing the statistics of the knowledge base integration result.
* <code>[./output/validated_inconsistencies.txt](./output/validated_inconsistencies.txt)</code>: Inconsistencies that has been resolved using the computational method and wet-lab validation.


