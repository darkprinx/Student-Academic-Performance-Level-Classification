# Students Academic Performance Level Classification

#### Visualization of students academic data & Classification of students level using different machine learning approaches

--------------------------------------------------------

### Inspiration 
-------------------------------
***The repository is developed to visualize students academic activities & other factors that impacts on students overall academic performance and also to classify students level based on their academic performance & others related informations. The datasets for the analysis are collected from [Students' Academic Performance Dataset](https://www.kaggle.com/aljarah/xAPI-Edu-Data)***

***The dataset file contains data of 480 students including 17 attributes.For the analysis purpose 10 attributes from the data file including the 'Class' attribute are taken.To apply different machine learning approaches for the classification,the total dataset is divided into 400 training set & 80 test set.***

**The whole description with all visualization & analisys can be found in this --> [NOTEBOOK](https://github.com/darkprinx/Students-Academic-Performance-Level-Classification/blob/master/Mechine%20learning.ipynb)**

----------------------------
### Dataset Information
-----------------------------
* ***Data Set Characteristics: Multivariate***

* ***Number of Instances: 480***

* ***Area: E-learning, Education, Predictive models, Educational Data Mining***

* ***Attribute Characteristics: Integer/Categorical***

* ***Number of Attributes: 10***

* ***Associated Tasks: Classification***

* ***Missing Values? No***

* ***File formats: xAPI-Edu-Data.csv***

----------------------------------
### Attributes
----------------------------------

1. ***Gender - Student's gender (Nominal: 'Male' or 'Female’)***

2. ***Educational Stages - Educational level student belongs (Nominal: ‘lowerlevel’,’MiddleSchool’,’HighSchool’)***

3. ***Section ID- Classroom student belongs (Nominal:’A’,’B’,’C’)***

4. ***Relation - Parent responsible for the student (Nominal:’Mum’,’Father’)***

5. ***Raised Hand - How many times the student raises his/her hand on the classroom (Numeric:0-100)***

6. ***Visited Resources - How many times the student visits a course content(numeric:0-100)***

7. ***Viewing Announcements - How many times the student checks the new announcements(numeric:0-100)***

8. ***Discussion Groups - How many times the student participate in discussion groups (numeric:0-100)***

9. ***Student Absence Days - The number of absence days for each student (nominal: above-7, under-7)***

10. ***Class - Overall performance level student belongs (Nominal: 'H','M','L')***

-----------------
### The students are classified into three numerical intervals based on their total grade/mark ####
----------------

   * ***Low-Level: interval includes values from 0 to 69***

   * ***Middle-Level: interval includes values from 70 to 89***

   * ***High-Level: interval includes values from 90-100***
   
------------------------

### Data Visualization
-------------------------------
***At first, the whole dataset is visualized in some different ways to make the reader understand the inner relations and dependencies among the attributes more clearly. This will also help to understand the factors those impacts on the students overall academic performance.***
