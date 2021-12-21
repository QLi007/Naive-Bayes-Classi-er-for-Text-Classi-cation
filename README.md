# Naive-Bayes-Classier-for-Text-Classication
ComS 573 Machine Learning  
1 Data Set  
The 20 Newsgroups data set is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. It was originally collected by Ken Lang,probably for his Newsweeder: Learning to filter netnews paper, though he did not explicitly mention this collection. The data is organized into 20 different newsgroups, each corresponding to a different topic. The original data set is available at http://qwone.com/~jason/20Newsgroups/.  
This processed version represents 18824 documents which have been divided into two subsets: training (11269 documents) and testing (7505 documents).
After unzipping the file, you will find six files: map.csv, train label.csv, train data.csv, test label.csv, test data.csv, vocabulary.txt. The vocabulary.txt contains all distinct words and other tokens in the 18824 documents. train data.csv and test data.csv are formatted "docIdx, wordIdx, count", where docIdx is the document id, wordIdx represents the word id (in correspondence to vocabulary.txt) and count is the frequency of the word in the document. train label.csv and test label.csv are simply a list of label id's indicating which newsgroup each document belongs to (with the row number representing the document id). The map.csv maps from label id's to label names.

![KU8`}N6II HJPE2_{OU6D05](https://user-images.githubusercontent.com/50083210/146994550-2d26bc61-7602-4be3-809e-0a511ceff745.png)
![image](https://user-images.githubusercontent.com/50083210/146994650-fd2ce6d0-3389-410f-a3fe-6b6096abf28d.png)
![image](https://user-images.githubusercontent.com/50083210/146994702-05029570-04da-4ba9-956a-8cc1cb4ae597.png)
