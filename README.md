# Multi-Task
Multi-Task net is a deep learning model for predicting protein-RNA binding interactions.

<br />
<p align="center">

  <h3 align="center">Multi-Task Net</h3>

  <p>
    Multi-Task net is a deep learning model for predicting protein-RNA binding interactions. This model was developed by Jonathan Karin and Hagai Michel as their bachelor degree final project, under the supervision of Dr.Yaron Orentein. <br />
  School of Electrical and Computer Engineering, Ben-Gurion University of the Negev,Beer-Sheva, Israel.
    <br />
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Bring the train and test data- in vitro](#Bring Data)
  * [Installation](#installation)
* [Usage](#usage)
* [Contact](#contact)



<!-- ABOUT THE PROJECT -->
## About The Project
..

### Built With
* [Python](https://www.python.org/) - 3.7 , Numpy and pandas libraries. 
* [Tensorflow](https://www.tensorflow.org/) - 2.0.0
* [Keras](https://keras.io/) - 2.3.1



<!-- GETTING STARTED -->
## Getting Started

Open folder and terminal for the project and write:
```sh
git clone https://github.com/JonathanKarin/Multi-Task/
cd Multi-Task
```

## Bring the train and test data- in vitro
Download the normalize RNACompete data in this  [Link](http://hugheslab.ccbr.utoronto.ca/supplementary-data/RNAcompete_eukarya/norm_data.txt.gz) from [RNAcompete](http://hugheslab.ccbr.utoronto.ca/supplementary-data/RNAcompete_eukarya/) site .
Download the secondary structure data from this [Link](https://drive.google.com/file/d/1jdDiR9LyWplZ7oFuccav9HlPngged9aH/view?usp=sharing).
Unzip the files to 'Data' folder.

## Prepare Data

```sh
python generate_data.py
```
## Train and Test
```sh
python train_and_test.py
```
The results will be saved as res.csv, the model would be saved in h5 format (model_41_9.h5).


<!-- CONTACT -->
## Contact

Dr. Yaron Orenstein - yaronore [at] bgu.ac.il
Jonathan Karin - karinjo [at ] post.bgu.ac.il


