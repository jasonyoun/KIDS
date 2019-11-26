# Knowledge Integration and Decision Support (KIDS)
KIDS is an end-to-end, automated framework for knowledge discovery consisting of three key components: 1) a knowledge graph constructor of biological  information, 2) an inconsistency resolver, and 3) a hypothesis generator.

![Figure 1](/images/Figure1.png)
*Figure 1. Overview of the KIDS framework.*

## Directories
* <code>[/kg_constructor](/kg_constructor)</code>: Code for creating the knowledge graph. Note that this directory also contains the source code for performing inconsistency resolution. Please refer to its own [README](/kg_constructor/README.md) file for more information.
* <code>[/hypothesis_generator](/hypothesis_generator)</code>: Code for generating the hypothesis based on the knowledge graph created. Please refer to its own [README](/hypothesis_generator/README.md) file for more information.

## Getting Started

### Dependencies
In addition to Python 3.6, following Python libraries are required.

```
argparse==1.1
configparser==3.5.0
imbalanced-learn==0.4.3
matplotlib==3.1.0
numpy==1.16.3
pandas==0.24.2
scikit-learn==0.20.0
scipy==1.3.1
tensorflow==1.13.1
```

You can install these required Python libraries as follows.

```
pip3 install -r requirements.txt
```

You will also need Java 7 or higher. If you are running Ubuntu 18.04, follow the steps below to install Java OpenJDK 11.

```
sudo apt update
sudo apt install openjdk-11-jdk
```

### Running
1. Construct the KG by following the [README](/kg_constructor/README.md) file.
2. Generate the hypothesis by following the [README](/hypothesis_generator/README.md) file.

### Docker
This section will be updated once we have the docker image.

## Citation
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut diam quam nulla porttitor massa id.

## Authors
* **Jason Youn** - *Some work* - <jyoun@ucdavis.edu>
* **Minseung Kim** - *Some work* - <msgkim@ucdavis.edu>
* **Nicholas Joodi** - *Some work* - <npjoodi@ucdavis.edu>

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments
* Hat tip to anyone whose code was used
* Inspiration
* etc
