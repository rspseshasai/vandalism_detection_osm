### Intro - OSM

OpenStreetMap (OSM) is a collaborative platform where volunteers worldwide contribute to continually enhancing the
world's most accessible map.

### Problem Statement

The openness of OSM also poses challenges, notably the proliferation of vandalism cases within user contributions. To
address this, this thesis aims to develop a real-time machine learning solution to flag potential vandalism in OSM
contributions as they occur.

### Proposed Thesis Framework

This repo consists of a master thesis code of a real-time ML vandalism notification system tailored to OSM's Kafka
stream. The aim is to dynamically analyzing user updates to detect anomalies indicative of potential vandalism in
real-time.

### Data

Data is stored in the link (WIP): https://drive.google.com/drive/folders/1EsxOFRJ5CNxpo3-TY91RqMdxeznnxQjJ?usp=sharing

### Test Cases

To run the test cases (in PyCharm): Mark the `test` folder as `test source root` and then set the `default test runner` to `pytest` in settings.
Then right-click on the test file (feature_extraction_tests.py) and run all. 