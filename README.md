Using Spark for Anomaly (Fraud) Detection
=========================================

Anomaly detection is a method used to detect outliers in a dataset and take some action. Example use cases can be detection of fraud in financial transactions, monitoring machines in a large server network, or finding faulty products in manufacturing.

This blog post explains the fundamentals of this Machine Learning algorithm and applies the logic on the Spark framework, in order to allow for large scale data processing:

http://micvog.com/2016/05/21/using-spark-for-anomaly-fraud-detection

Running on Spark
----------------

First compile:

```
sbt package
```

Run the tests:

```
sbt test
```

Download Spark (1.6.1) and put it in your Path from here: http://spark.apache.org/downloads.html

Run the project locally. From project root:

```
spark-submit --class "MainRun" --master local[8] target/scala-2.10/anomaly-detection_2.10-1.0.jar
```

