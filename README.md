# Road-Image-Classification

This is the README file for the final project of class 2110446 (2022/2) Data Science and Data Engineering.

## Project Members:

-   Yaninchaya Unjitwattana
-   Natnicha Coasol
-   Parin Opasphatikul
-   Nanthicha Makjinda

## Project Description:

The objective of this project is to classify road images based on their severity level of damage. We have categorized road damage into three levels: GOOD, POOR, and VERY POOR. The classification criteria are as follows:

### GOOD:

-   Rough surface
-   Small hole
-   Number of holes <= 1

### POOR:

-   Number of holes = 2
-   Medium width hole

### VERY POOR:

-   Number of holes >= 3
-   Wide hole
-   Hole with depth
-   Ponding

## Data Sources:

For this project, we have utilized data from the following sources:

-   TraffyFondue: We have labeled the road images ourselves.
-   Google: We have scraped additional road images from Google.
-   Kaggle: We have also obtained road image datasets from Kaggle.

## Technology Stack:

-   PyTorch: We have used PyTorch to build our classification model.
-   Selenium: We have utilized Selenium for web scraping road images.
-   MLflow: We have employed MLflow to track our model and deploy an endpoint for model inference on AWS.
-   PowerBI: Finally, we have visualized our results through PowerBI.
-   AWS: We have deployed our model on AWS.

![Project Diagram](images/diagram.png)
