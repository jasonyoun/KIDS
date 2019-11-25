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

## How to run

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

## Output files

### 4. out.txt
In this output file (a set of consistent knowledge), the knowledge is represented in triple format (Subject, Predicate, Object). There are 6 columns in the following order.
* Subject:
* Predicate:
* Object:
* Belief: the confidence score of this specific fact measured by the inconsistency corrector (default is AverageLog).
* Source size: number of sources supporting this knowledge
* Sources: the list of sources supporting this knowledge

### 5. inconsistency_out.txt
In this output file (a set of inconsistent knowledge), the knowledge is represented in triple format (Subject, Predicate, Object). The first six columns are for the fact with the highest belief (i.e. one that the inconsistency corrector think it is correct) among conflicting facts.
* Subject:
* Predicate:
* Object:
* Belief: the confidence score of this specific fact measured by the inconsistency corrector (default is AverageLog).
* Source size: number of sources supporting this knowledge
* Sources: the list of sources supporting this knowledge
* Total source size: the number of all sources (including the source size from the conflicting facts)
* Mean belief of conflicting tuples: the average of beliefs of all other tuples conflicting to one that is represented in the first 3 columns.
* Belief difference: the difference between the Belief and the Mean belief of conflicting tuples
* Conflicting tuple info: the list of all the tuples conflicting to one that is represented in the first 3 columns. It is the list represented in "[(element1), (element2), ...]" and each element of the list is represented by "tuple, sources, belief" where tuple is "(subject, predicate, object)", and source is "[source1, source2, ...]"
